import numpy as np
import torch
from TD7 import Agent

# https://arxiv.org/pdf/1405.5498
# Double Progressive Widening implementation

class Node:
    def __init__(self, zs, log_prob, parent=None, action=None):
        """
        Node in the search tree.

        Args:
            zs: State embedding (tensor)
            parent: Parent Node
            action: Action that led to this node from parent
        """
        self.zs = zs  # State embedding
        self.log_prob = log_prob  # Log probability of reaching this node
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
        return len(self.children) <= threshold

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
    def __init__(self, agent: Agent, alpha_a=0.5, alpha_s=0.5, k_a=2.0, k_s=2.0,
                 c_param=1.4, gamma=0.99, device=None):
        """
        Double Progressive Widening tree search using Agent's learned representations.

        Args:
            agent: TD7 Agent with encoder/decoder
            alpha_a: Progressive widening exponent for actions
            alpha_s: Progressive widening exponent for states
            k_a: Progressive widening constant for actions
            k_s: Progressive widening constant for states
            c_param: UCB exploration parameter
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
            log_prob: Log probability of the transition (if using nflow encoder)
        """
        with torch.no_grad():
            zs_batch = zs.unsqueeze(0) if zs.dim() == 1 else zs
            action_batch = action.unsqueeze(0) if action.dim() == 1 else action

            next_zs = self.agent.encoder.zsa(zs_batch, action_batch)

            # Calculate log probability if using NFlow encoder
            log_prob = 0.0
            if hasattr(self.agent.encoder, 'flow'):
                context = torch.cat([zs_batch, action_batch], dim=-1)
                log_prob = float(self.agent.encoder.flow.log_prob(next_zs, context=context).item())

        return next_zs.squeeze(0), log_prob

    def get_value_estimate(self, parent_zs, action, child_zs):
        """
        Get Q-value estimate for state-action pair.

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

            # Critic needs raw state, but we can decode it
            state = self.agent.decoder.decode_state(parent_zs_batch)

            Q = self.agent.critic(state, action_batch, child_zs_batch, parent_zs_batch)
            value = Q.min(dim=1)[0]  # Take min of two Q-values (conservative)

        return float(value.item())

    def search(self, root_state, num_simulations=100):
        """
        Perform tree search from root state.

        Args:
            root_state: Initial state (raw state, not embedding)
            num_simulations: Number of MCTS simulations to run

        Returns:
            best_action: The best action to take from root
        """
        # Convert root state to embedding
        with torch.no_grad():
            if isinstance(root_state, np.ndarray):
                root_state = torch.tensor(root_state, dtype=torch.float32, device=self.device)
            root_state = root_state.unsqueeze(0) if root_state.dim() == 1 else root_state
            root_zs = self.agent.encoder.zs(root_state).squeeze(0)

        root = Node(root_zs, log_prob=0.0)

        for sim in range(num_simulations):
            node = root
            search_path = []

            # Selection: Navigate down the tree using UCB
            while node.action_samples and not node.should_expand_actions(self.alpha_a, self.k_a):
                action_idx, action = node.select_action(self.c_param)

                if action_idx is None:
                    break

                # Check if this action has been explored
                if action_idx not in node.children:
                    # Create new child node
                    next_zs, log_prob = self.get_next_zs(node.zs, action)
                    child = Node(next_zs, log_prob=log_prob, parent=node, action=action)
                    node.children[action_idx] = child

                    # Evaluate this transition
                    value = self.get_value_estimate(node.zs, action, next_zs)
                    search_path.append((node, child, value))
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
                next_zs, log_prob = self.get_next_zs(node.zs, new_action)
                child = Node(next_zs, log_prob=log_prob, parent=node, action=new_action)
                node.children[action_idx] = child

                # Evaluate this transition
                value = self.get_value_estimate(node.zs, new_action, next_zs)
                search_path.append((node, child, value))

            # Backpropagation: Update values along the path
            if search_path:
                for parent_node, child_node, transition_value in search_path:
                    # Update both parent and child
                    parent_node.visits += 1
                    parent_node.value_sum += transition_value
                    child_node.visits += 1
                    child_node.value_sum += transition_value * self.gamma

        # Select best action from root based on average value (not just visit count)
        if not root.action_samples:
            # Fallback: sample action without noise
            return self.sample_action(root_zs, add_noise=False).cpu().numpy()

        # Choose action with best average value
        best_idx = max(root.children.keys(),
                      key=lambda idx: root.children[idx].get_value())
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


if __name__ == "__main__":
    import argparse
    import gymnasium as gym

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                       default="results/nflow_HalfCheetah-v5_envaction_seed_0_1000000/models/final_model.pt",
                       help="Path to the trained model")
    parser.add_argument("--env", type=str, default="HalfCheetah-v5",
                       help="Environment name")
    parser.add_argument("--num_simulations", type=int, default=100,
                       help="Number of MCTS simulations per action")
    parser.add_argument("--num_episodes", type=int, default=5,
                       help="Number of episodes to run")
    parser.add_argument("--alpha_a", type=float, default=0.5,
                       help="Progressive widening exponent for actions")
    parser.add_argument("--alpha_s", type=float, default=0.5,
                       help="Progressive widening exponent for states")
    parser.add_argument("--k_a", type=float, default=2.0,
                       help="Progressive widening constant for actions")
    parser.add_argument("--k_s", type=float, default=2.0,
                       help="Progressive widening constant for states")
    parser.add_argument("--c_param", type=float, default=1.4,
                       help="UCB exploration parameter")
    parser.add_argument("--render", action="store_true",
                       help="Render the environment")
    args = parser.parse_args()

    print("="*60)
    print("Double Progressive Widening Tree Search")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Environment: {args.env}")
    print(f"Simulations per action: {args.num_simulations}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Parameters: alpha_a={args.alpha_a}, alpha_s={args.alpha_s}, k_a={args.k_a}, k_s={args.k_s}, c={args.c_param}")
    print("="*60)

    # Load the trained agent
    print(f"\nLoading agent from {args.model_path}...")
    agent = Agent.load(args.model_path)
    agent.encoder.eval()
    agent.decoder.eval()
    agent.actor.eval()
    agent.critic.eval()
    print("Agent loaded successfully!")

    # Create environment
    render_mode = "human" if args.render else "rgb_array"
    env = gym.make(args.env, render_mode=render_mode)

    # Create tree search
    tree_search = DoubleProgressiveWidening(
        agent=agent,
        alpha_a=args.alpha_a,
        alpha_s=args.alpha_s,
        k_a=args.k_a,
        k_s=args.k_s,
        c_param=args.c_param,
        gamma=0.99
    )

    print(f"\nTree search initialized with device: {agent.device}")
    print("\nRunning episodes with tree search...\n")

    # Run episodes
    episode_rewards = []

    for ep in range(args.num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        step = 0

        while not done:
            # Use tree search to select action
            action = tree_search.search(state, num_simulations=args.num_simulations)

            # Take action in environment
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            step += 1

            if args.render:
                env.render()

        episode_rewards.append(episode_reward)
        print(f"Episode {ep+1}/{args.num_episodes}: Reward = {episode_reward:.2f}, Steps = {step}")

    print("\n" + "="*60)
    print("Results")
    print("="*60)
    print(f"Mean reward: {np.mean(episode_rewards):.2f}")
    print(f"Std reward: {np.std(episode_rewards):.2f}")
    print(f"Min reward: {np.min(episode_rewards):.2f}")
    print(f"Max reward: {np.max(episode_rewards):.2f}")
    print("="*60)

    # Comparison: Run same episodes without tree search (just using actor)
    print("\n" + "="*60)
    print("Baseline Comparison (Actor only, no tree search)")
    print("="*60)

    baseline_rewards = []
    for ep in range(args.num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        step = 0

        while not done:
            # Use actor directly without tree search
            action = agent.select_action(state, use_checkpoint=False, use_exploration=False)

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            step += 1

        baseline_rewards.append(episode_reward)
        print(f"Episode {ep+1}/{args.num_episodes}: Reward = {episode_reward:.2f}, Steps = {step}")

    print("\n" + "="*60)
    print("Baseline Results")
    print("="*60)
    print(f"Mean reward: {np.mean(baseline_rewards):.2f}")
    print(f"Std reward: {np.std(baseline_rewards):.2f}")
    print(f"Min reward: {np.min(baseline_rewards):.2f}")
    print(f"Max reward: {np.max(baseline_rewards):.2f}")
    print("="*60)

    print("\n" + "="*60)
    print("Improvement")
    print("="*60)
    improvement = np.mean(episode_rewards) - np.mean(baseline_rewards)
    print(f"Mean reward improvement: {improvement:+.2f} ({improvement/np.mean(baseline_rewards)*100:+.2f}%)")
    print("="*60)

    env.close()
