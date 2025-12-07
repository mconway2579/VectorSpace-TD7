import numpy as np
import torch
from TD7 import Agent
import argparse
import gymnasium as gym

# https://arxiv.org/pdf/1405.5498
# Double Progressive Widening implementation

class Node:
    def __init__(self, zs, parent=None, action=None):
        """
        Node in the search tree.

        Args:
            zs: State embedding (tensor)
            parent: Parent Node
            action: Action that led to this node from parent
        """
        self.zs = zs  # State embedding
        self.parent = parent
        self.action = action  # Action that led to this node
        self.children = {}  # Maps action index to child Node
        self.action_samples = []  # List of sampled actions
        self.visits = 0
        self.value_sum = 0.0

    def get_value(self):
        """Average value of this node."""
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def ucb(self, c_param=1.4, parent_visits=None):
        """
        UCB1 formula for action selection.

        Args:
            c_param: Exploration parameter
            parent_visits: Number of visits to parent node
        """
        if self.visits == 0:
            return float('inf')

        if parent_visits is None:
            parent_visits = self.parent.visits if self.parent else 1

        exploitation = self.get_value()
        exploration = c_param * np.sqrt(np.log(parent_visits) / self.visits)
        return exploitation + exploration

    def should_expand_actions(self, alpha=0.5, k_a=2.0):
        """
        Progressive widening for action space.
        Returns True if we should sample a new action.

        Formula: |A(s)| <= k_a * visits^alpha
        """
        num_actions = len(self.action_samples)
        threshold = k_a * (self.visits ** alpha)
        return num_actions <= threshold

    def should_expand_states(self, action_idx, alpha=0.5, k_s=2.0):
        """
        Progressive widening for state space.
        Returns True if we should sample a new state transition.

        Formula: |S(s,a)| <= k_s * visits(s,a)^alpha
        """
        if action_idx not in self.children:
            return True
        child = self.children[action_idx]
        # In simple version, each action leads to one state
        # For stochastic environments, you'd track multiple children per action
        threshold = k_s * (child.visits ** alpha)
        # Should check states for this specific action, but in deterministic case it's always 1
        return 1 <= threshold

    def select_action(self, c_param=1.4):
        """
        Select the best action according to UCB.

        Returns:
            action_idx: Index of the selected action
            action: The action tensor
        """
        if not self.action_samples:
            return None, None

        best_idx = -1
        best_score = -float('inf')

        for idx in range(len(self.action_samples)):
            if idx in self.children:
                score = self.children[idx].ucb(c_param, self.visits)
            else:
                score = float('inf')  # Unvisited actions have highest priority

            if score > best_score:
                best_score = score
                best_idx = idx

        return best_idx, self.action_samples[best_idx]

    def backpropagate(self, value):
        """
        Update the node's value and visit count, propagating up the tree.

        Args:
            value: The value to backpropagate
        """
        node = self
        while node is not None:
            node.visits += 1
            node.value_sum += value
            node = node.parent


class DoubleProgressiveWidening:
    def __init__(self, agent: Agent, num_simulations, alpha_a=0.25, alpha_s=0.25, k_a=1.0, k_s=1.0,
                 c_param=0.5, gamma=0.99, device=None):
        """
        Double Progressive Widening tree search using Agent's learned representations.

        Default hyperparameters are set aggressively for speed:
        - alpha_a=0.25, alpha_s=0.25: Very restrictive progressive widening
        - k_a=1.0, k_s=1.0: Minimal action/state branching
        - c_param=0.5: Low exploration, high exploitation
        These settings prioritize speed over thoroughness.

        Args:
            agent: TD7 Agent with encoder/decoder
            alpha_a: Progressive widening exponent for actions (lower = more restrictive)
            alpha_s: Progressive widening exponent for states (lower = more restrictive)
            k_a: Progressive widening constant for actions (lower = fewer actions)
            k_s: Progressive widening constant for states (lower = fewer states)
            c_param: UCB exploration parameter (lower = more exploitation)
            gamma: Discount factor
            device: torch device
        """
        self.agent = agent
        self.alpha_a = alpha_a
        self.alpha_s = alpha_s
        self.k_a = k_a
        self.k_s = k_s
        self.c_param = c_param
        self.gamma = gamma
        self.device = device if device else agent.device
        self.num_simulations = num_simulations

    def sample_action(self, zs, add_noise=False):
        """
        Sample an action using the agent's policy.

        Args:
            zs: State embedding
            add_noise: Whether to add exploration noise (for diversity in tree)

        Returns:
            action: Sampled action tensor
        """
        with torch.no_grad():
            # Use actor to generate action
            state_encoding_batch = zs.unsqueeze(0) if zs.dim() == 1 else zs
            state_batch = self.agent.decoder.decode_state(state_encoding_batch)
            action = self.agent.actor(state_batch, state_encoding_batch)  # Actor uses decoded state

            # Add small exploration noise only if requested (for action diversity)
            if add_noise:
                action = action + torch.randn_like(action) * 0.05
                action = action.clamp(-1, 1)

        return action.squeeze(0)

    def get_next_zs(self, zs, action):
        """
        Get next state embedding using encoder's transition model.

        Args:
            zs: Current state embedding
            action: Action tensor

        Returns:
            next_zs: Next state embedding predicted by zsa
        """
        with torch.no_grad():
            zs_batch = zs.unsqueeze(0) if zs.dim() == 1 else zs
            action_batch = action.unsqueeze(0) if action.dim() == 1 else action

            next_zs = self.agent.encoder.zsa(zs_batch, action_batch)

        return next_zs.squeeze(0)

    def get_value_estimate(self, parent_zs, action, child_zs):
        """
        Get Q-value estimate for state-action pair.

        Uses critic directly which already predicts Q(s,a) = r + gamma*V(s').
        This is much faster than decomposing into reward + next_value.

        Args:
            parent_zs: Parent state embedding
            action: Action tensor
            child_zs: Next state embedding (from zsa)

        Returns:
            value: Estimated Q-value
        """
        with torch.no_grad():
            parent_zs_batch = parent_zs.unsqueeze(0) if parent_zs.dim() == 1 else parent_zs
            action_batch = action.unsqueeze(0) if action.dim() == 1 else action
            child_zs_batch = child_zs.unsqueeze(0) if child_zs.dim() == 1 else child_zs

            # Decode state only once (decoder is fast)
            state = self.agent.decoder.decode_state(parent_zs_batch)

            # Get Q-value directly from critic (already includes reward + future value)
            Q = self.agent.critic(state, action_batch, child_zs_batch, parent_zs_batch)
            value = Q.min(dim=1)[0]  # Take min of two Q-values (conservative)

        return float(value.item())

    def select_action(self, root_state):
        """
        Perform tree search from root state and return best action.

        Args:
            root_state: Initial state (raw state, not embedding)

        Returns:
            best_action: The best action to take from root (numpy array)
        """
        # Convert root state to embedding
        with torch.no_grad():
            if isinstance(root_state, np.ndarray):
                root_state = torch.tensor(root_state, dtype=torch.float32, device=self.device)
            root_state = root_state.unsqueeze(0) if root_state.dim() == 1 else root_state
            root_zs = self.agent.encoder.zs(root_state).squeeze(0)

        root = Node(root_zs)

        for sim in range(self.num_simulations):
            node = root
            search_path = []
            leaf_node = None
            leaf_value = None

            # Selection: Navigate down the tree using UCB
            while node.action_samples and not node.should_expand_actions(self.alpha_a, self.k_a):
                action_idx, action = node.select_action(self.c_param)

                if action_idx is None:
                    break

                # Check if this action has been explored
                if action_idx not in node.children:
                    # Create new child node
                    next_zs = self.get_next_zs(node.zs, action)
                    child = Node(next_zs, parent=node, action=action)
                    node.children[action_idx] = child

                    # Evaluate this transition
                    value = self.get_value_estimate(node.zs, action, next_zs)
                    search_path.append((node, child, value))
                    leaf_node = child
                    leaf_value = value
                    break
                else:
                    child = node.children[action_idx]
                    node = child

            # Expansion: Add new action if progressive widening allows
            if not search_path and node.should_expand_actions(self.alpha_a, self.k_a):
                # Sample action with small noise for diversity
                new_action = self.sample_action(node.zs, add_noise=(sim > 0))
                action_idx = len(node.action_samples)
                node.action_samples.append(new_action)

                # Create child node for this new action
                next_zs = self.get_next_zs(node.zs, new_action)
                child = Node(next_zs, parent=node, action=new_action)
                node.children[action_idx] = child

                # Evaluate this transition
                value = self.get_value_estimate(node.zs, new_action, next_zs)
                search_path.append((node, child, value))
                leaf_node = child
                leaf_value = value

            # If we reached an existing leaf without expansion, evaluate it
            if leaf_node is None and node != root:
                # We navigated to an existing leaf - evaluate it
                leaf_node = node
                # Re-evaluate the leaf's value (could also use cached value)
                if node.parent is not None:
                    leaf_value = self.get_value_estimate(node.parent.zs, node.action, node.zs)
                else:
                    # Shouldn't happen, but fallback
                    leaf_value = node.get_value()

            # Backpropagation: Update values from leaf to root
            if leaf_node is not None:
                leaf_node.backpropagate(leaf_value)

        # Select best action from root based on visit count (robust child)
        if not root.action_samples:
            # Fallback: sample action without noise
            return self.sample_action(root_zs, add_noise=False).cpu().numpy()

        # Choose most visited action (robust child - standard MCTS)
        best_idx = max(root.children.keys(),
                      key=lambda idx: root.children[idx].visits)
        best_action = root.action_samples[best_idx]

        return best_action.cpu().numpy()

    def get_statistics(self, root):
        """
        Get statistics about the search tree.

        Args:
            root: Root node of the tree

        Returns:
            dict: Statistics about the tree
        """
        total_nodes = 0
        max_depth = 0

        def traverse(node, depth=0):
            nonlocal total_nodes, max_depth
            total_nodes += 1
            max_depth = max(max_depth, depth)

            for child in node.children.values():
                traverse(child, depth + 1)

        traverse(root)

        return {
            'total_nodes': total_nodes,
            'max_depth': max_depth,
            'root_visits': root.visits,
            'root_num_actions': len(root.action_samples),
            'root_num_children': len(root.children)
        }

class ActorWrapper:
    """Wrapper for Agent to provide compatible select_action interface."""
    def __init__(self, agent):
        self.agent = agent

    def select_action(self, state):
        """
        Select action using agent's policy without tree search.

        Args:
            state: Current state (numpy array)

        Returns:
            action: Selected action (numpy array)
        """
        return self.agent.select_action(state, use_checkpoint=False, use_exploration=False)


def eval_method(action_model, env, args):
    """
    Evaluate an action model (either ActorWrapper or DoubleProgressiveWidening).

    Args:
        action_model: Object with select_action(state) method
        env: Gymnasium environment
        args: Arguments object with num_episodes, render

    Returns:
        dict: Results with mean, std, min, max, raw episode rewards
    """
    episode_rewards = []

    for ep in range(args.num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        step = 0

        while not done:
            # Select action using the provided model
            action = action_model.select_action(state)

            # Take action in environment
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            step += 1

            if args.render:
                env.render()

        episode_rewards.append(episode_reward)
        print(f"Episode {ep+1}/{args.num_episodes}: Reward = {episode_reward:.2f}, Steps = {step}")

    results = {
        "mean": np.mean(episode_rewards),
        "std": np.std(episode_rewards),
        "min": np.min(episode_rewards),
        "max": np.max(episode_rewards),
        "raw": episode_rewards
    }
    print("\n")
    print("="*60)
    print("Results")
    print("="*60)
    print(f"Mean reward: {results['mean']:.2f}")
    print(f"Std reward: {results['std']:.2f}")
    print(f"Min reward: {results['min']:.2f}")
    print(f"Max reward: {results['max']:.2f}")
    print("="*60)
    return results



if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_paths", type=str, nargs='+',
                       default=[
                           "/home/max/VectorSpace-TD7/results/nflow_HalfCheetah-v5_DeterministicActor_seed_0_1000000/models/final_model.pt",
                           "/home/max/VectorSpace-TD7/results/nflow_HalfCheetah-v5_ProbabilisticActor_seed_0_1000000/models/final_model.pt",
                           "/home/max/VectorSpace-TD7/results/td7_HalfCheetah-v5_DeterministicActor_seed_0_1000000/models/final_model.pt",
                           "/home/max/VectorSpace-TD7/results/td7_HalfCheetah-v5_ProbabilisticActor_seed_0_1000000/models/final_model.pt"
                       ],
                       help="List of paths to trained models")
    parser.add_argument("--env", type=str, default="HalfCheetah-v5",
                       help="Environment name")
    parser.add_argument("--num_simulations", type=int, default=20,
                       help="Number of MCTS simulations per action (aggressive default: 20)")
    parser.add_argument("--num_episodes", type=int, default=5,
                       help="Number of episodes to run")
    parser.add_argument("--alpha_a", type=float, default=0.25,
                       help="Progressive widening exponent for actions (aggressive default: 0.25)")
    parser.add_argument("--alpha_s", type=float, default=0.25,
                       help="Progressive widening exponent for states (aggressive default: 0.25)")
    parser.add_argument("--k_a", type=float, default=1.0,
                       help="Progressive widening constant for actions (aggressive default: 1.0)")
    parser.add_argument("--k_s", type=float, default=1.0,
                       help="Progressive widening constant for states (aggressive default: 1.0)")
    parser.add_argument("--c_param", type=float, default=0.5,
                       help="UCB exploration parameter (aggressive default: 0.5)")
    parser.add_argument("--render", action="store_true",
                       help="Render the environment")
    args = parser.parse_args()

    # Create environment once
    render_mode = "human" if args.render else "rgb_array"
    env = gym.make(args.env, render_mode=render_mode)

    # Store all results
    all_results = []

    # Loop over all model paths
    for model_idx, model_path in enumerate(args.model_paths):
        print("\n" + "="*80)
        print(f"EVALUATING MODEL {model_idx + 1}/{len(args.model_paths)}")
        print("="*80)
        print(f"Model: {model_path}")
        print(f"Environment: {args.env}")
        print(f"Simulations per action: {args.num_simulations}")
        print(f"Episodes: {args.num_episodes}")
        print(f"Parameters: alpha_a={args.alpha_a}, alpha_s={args.alpha_s}, k_a={args.k_a}, k_s={args.k_s}, c={args.c_param}")
        print("="*80)

        # Load the trained agent
        print(f"\nLoading agent from {model_path}...")
        agent = Agent.load(model_path)
        agent.set_device("cpu")

        # Set all models to eval mode for faster inference
        agent.encoder.eval()
        agent.decoder.eval()
        agent.actor.eval()
        agent.critic.eval()
        agent.reward_predictor.eval()

        print("Agent loaded successfully!")

        # Create tree search
        tree_search = DoubleProgressiveWidening(
            agent=agent,
            num_simulations=args.num_simulations,
            alpha_a=args.alpha_a,
            alpha_s=args.alpha_s,
            k_a=args.k_a,
            k_s=args.k_s,
            c_param=args.c_param,
            gamma=0.99,
        )
        print(f"\nTree search initialized with device: {agent.device}")
        print("\nRunning episodes with tree search...\n")
        ts_results = eval_method(tree_search, env, args)

        # Comparison: Run same episodes without tree search (just using actor)
        print("\n" + "="*60)
        print("Baseline Comparison (Actor only, no tree search)")
        print("="*60)

        # Wrap agent in ActorWrapper for compatible interface
        actor_baseline = ActorWrapper(agent)
        bs_results = eval_method(actor_baseline, env, args)

        print("\n" + "="*60)
        print("Improvement")
        print("="*60)
        improvement = ts_results["mean"] - bs_results["mean"]
        print(f"Mean reward improvement: {improvement:+.2f} ({improvement/bs_results['mean']*100:+.2f}%)")
        print("="*60)

        # Store results
        all_results.append({
            "model_path": model_path,
            "tree_search": ts_results,
            "baseline": bs_results,
            "improvement": improvement,
            "improvement_pct": (improvement / bs_results['mean'] * 100)
        })

    env.close()

    # Print summary of all results
    print("\n\n" + "="*80)
    print("SUMMARY OF ALL MODELS")
    print("="*80)
    for idx, result in enumerate(all_results):
        model_name = result["model_path"].split("/")[-3] if "/" in result["model_path"] else result["model_path"]
        print(f"\nModel {idx + 1}: {model_name}")
        print(f"  Tree Search: {result['tree_search']['mean']:.2f} ± {result['tree_search']['std']:.2f}")
        print(f"  Baseline:    {result['baseline']['mean']:.2f} ± {result['baseline']['std']:.2f}")
        print(f"  Improvement: {result['improvement']:+.2f} ({result['improvement_pct']:+.2f}%)")
    print("="*80)
