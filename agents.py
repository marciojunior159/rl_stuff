import numpy as np

def rand_argmax(q_values, actions):
        """
        Special argmax function to randomly selected
        between array positions of similar (max) value

        Parameters
        ----------
        i : int
            position of selected action 
        reward : numeric
            feedback received from the environment
        alpha : 0.1, optional
            step-size of the incremental update

        Returns
        -------
        integer
            the action to be taken by the agent

        """

        return np.random.choice(actions[q_values == np.max(q_values)])


class Greedy:
    """
    Greedy agent standard class

    ...

    Attributes
    ----------
    env : kArmedBandits
        The environment that the agent is part of
    actions : list
        List of possible actions taken by the agent
    q_values : list, optional
        List of (initial) expected rewards for each action

    Methods
    -------
    update_q(i, reward, alpha=0.1)
        Updates expected q based on env response
    act()
        Agent takes an action

    """
    def __init__(self, env, actions, q_values=None):
        self.actions = actions
        self.actcount = [0 for _ in actions]
        self.env = env
        self.last_reward = 0

        if q_values is None:
            self.q_values = [0 for _ in actions]
        else:
            self.q_values = q_values

    def update_q(self, i, reward, alpha=None):
        """
        Updates expected q based on env response.
        Incremental Q: Qn+1 = Qn + α*(Rn - Qn)

        Parameters
        ----------
        i : int
            position of selected action 
        reward : numeric
            feedback received from the environment
        alpha : 0.1, optional
            step-size of the incremental update
            
        """
        if alpha == None: 
            if self.actcount[i] == 0: alpha = 1
            else: alpha = 1./self.actcount[i]
        self.q_values[i] += alpha*(reward - self.q_values[i])

    def act(self):
        """
        Agent takes an action, gets the reward and
        updates Q
        """
        current_action = rand_argmax(self.q_values, self.actions)
        reward = self.env.get_reward(current_action)
        self.last_reward = reward
        self.last_action = current_action
        self.update_q(current_action, reward, alpha=0.1)
        self.actcount[current_action] += 1
    

class EpsilonGreedy(Greedy):
    """
    ε-Greedy agent class

    ...

    Attributes
    ----------
    env : kArmedBandits
        The environment that the agent is part of
    actions : list
        List of possible actions taken by the agent
    epsilon : float
        Variable defining the level of exploration
    q_values : list, optional
        List of (initial) expected rewards for each action

    Methods
    -------
    act()
        Agent takes an action

    """

    def __init__(self, env, actions, epsilon, q_values=None):
        super().__init__(env, actions, q_values)
        self.epsilon = epsilon

    def act(self):
        """
        Agent takes an action, sometimes randomly, if the
        generated random number is smaller than ε, then it
        gets the reward of the action and updates Q
        """
        if np.random.random() < self.epsilon:
            current_action = np.random.choice(self.actions)
        else:
            current_action = rand_argmax(self.q_values, self.actions)
        reward = self.env.get_reward(current_action)
        self.last_reward = reward
        self.last_action = current_action
        self.update_q(current_action, reward, alpha=0.1)
        

class UpperConfidenceBound(Greedy):
    """
    UCB agent class

    ...

    Attributes
    ----------
    env : kArmedBandits
        The environment that the agent is part of
    actions : list
        List of possible actions taken by the agent
    c : float
        Variable defining the level of exploration 
    q_values : list, optional
        List of (initial) expected rewards for each action

    Methods
    -------
    ucb_argmax()
        Special Argmax calc for UCB
    act()
        Agent takes an action

    """

    def __init__(self, env, actions, c, q_values=None):
        super().__init__(env, actions, q_values)
        self.c = c

    def ucb_argmax(self):
        """
        Special argmax function to randomly selected
        between array positions of similar (max) value

        Parameters
        ----------
        i : int
            position of selected action 
        reward : numeric
            feedback received from the environment
        alpha : 0.1, optional
            step-size of the incremental update

        Returns
        -------
        integer
            the action to be taken by the agent

        """
        special_q = np.array([self.q_values[a] + \
                              self.c*np.sqrt(np.log(np.sum(self.actcount))/self.actcount[a]) \
                              if self.actcount[a] > 0 \
                              else np.Inf for a in self.actions])
        return np.random.choice(self.actions[special_q == np.max(special_q)])
        
    def act(self):
        """
        Agent takes an action, gets the reward and
        updates Q
        """
        current_action = self.ucb_argmax()
        reward = self.env.get_reward(current_action)
        self.last_reward = reward
        self.last_action = current_action
        self.update_q(current_action, reward, alpha=0.1)
        self.actcount[current_action] += 1

