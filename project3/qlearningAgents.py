# qlearningAgents.py
# ------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
  """
    Q-Learning Agent

    Functions you should fill in:
      - getQValue
      - getAction
      - getValue
      - getPolicy
      - update

    Instance variables you have access to
      - self.epsilon (exploration prob)
      - self.alpha (learning rate)
      - self.discount (discount rate)

    Functions you should use
      - self.getLegalActions(state)
        which returns legal actions
        for a state
  """
  def __init__(self, **args):
    "You can initialize Q-values here..."
    ReinforcementAgent.__init__(self, **args)

    "*** YOUR CODE HERE ***"
    self.Qvalue = util.Counter()





  def getQValue(self, state, action):
    """
      Returns Q(state,action)
      Should return 0.0 if we never seen
      a state or (state,action) tuple
    """
    "*** YOUR CODE HERE ***"
    #return self.QValue[(state,action)]
    return self.Qvalue[(state, action)]

  def getValue(self, state):
    """
      Returns max_action Q(state,action)
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
    "*** YOUR CODE HERE ***"
    #if(len(self.getLegalActions(state))==0):
    #  return 0
    #myaction = self.getLegalActions(state);
    #nxtState = [];
    #for x in myaction:
    #  nxtState.append(self.getQValue(state,x));
    #return max(nxtState);
    myQvalues = []
    if len(self.getLegalActions(state)) == 0:
            return 0.0
    actions = self.getLegalActions(state)
    for a in actions:
        myQvalues.append(self.getQValue(state, a))
    return max(myQvalues)



  def getPolicy(self, state):
    """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """
    "*** YOUR CODE HERE ***"
    #if(len(self.getLegalActions(state))==0):
    #  return None
    #Bestaction = None;
    #MaxQ = 0;
    #myaction = self.getLegalActions(state);
    #for x in myaction:
    #  TempQ = self.getQValue(state,x);
    #  if TempQ > MaxQ or Bestaction is None:
    #    MaxQ = TempQ;
    #    Bestaction = x;
    #return Bestaction

    Qvalues = util.Counter()
    if len(self.getLegalActions(state)) == 0:
        return None
    actions = self.getLegalActions(state)
    for a in actions:
        Qvalues[(state, a)] = self.getQValue(state, a)
    Qvalues.sortedKeys()
    for q in Qvalues.keys():
        if not q == Qvalues.argMax():
            del Qvalues[q]
    policy = [q[1] for q in Qvalues.keys()]
    Bestaction = random.choice(policy);
    return Bestaction



  def getAction(self, state):
    """
      Compute the action to take in the current state.  With
      probability self.epsilon, we should take a random action and
      take the best policy action otherwise.  Note that if there are
      no legal actions, which is the case at the terminal state, you
      should choose None as the action.

      HINT: You might want to use util.flipCoin(prob)
      HINT: To pick randomly from a list, use random.choice(list)
    """
    # Pick Action
    legalActions = self.getLegalActions(state)
    action = None
    "*** YOUR CODE HERE ***"
    if len(self.getLegalActions(state)) == 0:
            return action
    prob = self.epsilon;
    if util.flipCoin(prob):
        action = random.choice(legalActions)
    else:
        action = self.getPolicy(state)
    return action


  def update(self, state, action, nextState, reward):
    """
      The parent class calls this to observe a
      state = action => nextState and reward transition.
      You should do your Q-Value update here

      NOTE: You should never call this function,
      it will be called on your behalf
    """
    "*** YOUR CODE HERE ***"
    #X1 = (1 - self.alpha) * self.getQValue(state, action)
    #action = self.getLegalActions(nextState)
    #nextQvalue = [self.QValue[(nextState, a)] for a in action]
    #if len(nextQvalue) == 0:
    #  X2 = reward;
    #else:
    #  X2 = reward + (self.discount * max(nextQvalue))
    #X3 = self.alpha * X2;
    #self.QValue[(state,action)] = X1 + X3;

    actions = self.getLegalActions(nextState)
    nxtState = [self.Qvalue[(nextState, a)] for a in actions]
    if not len(nxtState):
        sample = reward
    else:
        sample=reward+self.discount*max(nxtState)
    myQ=(1.0-self.alpha)*self.getQValue(state, action)+self.alpha*sample
    self.Qvalue[state,action] = myQ


class PacmanQAgent(QLearningAgent):
  "Exactly the same as QLearningAgent, but with different default parameters"

  def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
    """
    These default parameters can be changed from the pacman.py command line.
    For example, to change the exploration rate, try:
        python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

    alpha    - learning rate
    epsilon  - exploration rate
    gamma    - discount factor
    numTraining - number of training episodes, i.e. no learning after these many episodes
    """
    args['epsilon'] = epsilon
    args['gamma'] = gamma
    args['alpha'] = alpha
    args['numTraining'] = numTraining
    self.index = 0  # This is always Pacman
    QLearningAgent.__init__(self, **args)

  def getAction(self, state):
    """
    Simply calls the getAction method of QLearningAgent and then
    informs parent of action for Pacman.  Do not change or remove this
    method.
    """
    action = QLearningAgent.getAction(self,state)
    self.doAction(state,action)
    return action


class ApproximateQAgent(PacmanQAgent):
  """
     ApproximateQLearningAgent

     You should only have to overwrite getQValue
     and update.  All other QLearningAgent functions
     should work as is.
  """
  def __init__(self, extractor='IdentityExtractor', **args):
    self.featExtractor = util.lookup(extractor, globals())()
    PacmanQAgent.__init__(self, **args)

    # You might want to initialize weights here.
    "*** YOUR CODE HERE ***"
    self.weight = util.Counter()

  def getQValue(self, state, action):
    """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator
    """
    "*** YOUR CODE HERE ***"
    myQval = 0.0
    fVector = self.featExtractor.getFeatures(state, action)
    for a in fVector.keys():
        myQval = self.weight[a] * fVector[a] +myQval
    return myQval

  def update(self, state, action, nextState, reward):
    """
       Should update your weights based on transition
    """
    "*** YOUR CODE HERE ***"
    fVector = self.featExtractor.getFeatures(state, action)
    for a in fVector.keys():
        self.weight[a] = self.alpha*(reward + self.discount*self.getValue(nextState) - self.getQValue(state, action))*fVector[a] +self.weight[a]


  def final(self, state):
    "Called at the end of each game."
    # call the super-class final method
    PacmanQAgent.final(self, state)

    # did we finish training?
    if self.episodesSoFar == self.numTraining:
      # you might want to print your weights here for debugging
      "*** YOUR CODE HERE ***"
      print(self.weight)

      pass
