# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """


  def getAction(self, gameState):
    """
    You do not need to change this method, but you're welcome to.

    getAction chooses among the best options according to the evaluation function.

    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    "Add more of your code here if you want to"

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    Design a better evaluation function here.

    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (newFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    my_score = 0

    if action == 'stop':
      return -1000000


    # if newScaredTimes == 0:
    #   return -1000000

    for state in newGhostStates:
      if state.getPosition == currentGameState.getPacmanPosition() and (newScaredTimes == 0):
        return -1000000

    ghostDistance = 0
    temp_d = 1000000000
    for state in newGhostStates:
      temp_d = manhattanDistance(newPos, state.getPosition())
      if temp_d > ghostDistance:
        ghostDistance = temp_d

    
    
    food = currentGameState.getFood().asList()
    cloestFood = manhattanDistance(food[0],newPos)
    distance = 0
    for x in food:
      distance = manhattanDistance(x, newPos)
      if distance < cloestFood:
        cloestFood = distance

    my_score = ghostDistance - cloestFood*2

    return my_score


def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)



class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (question 2)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
    Solution = []
    testVal = 0
    #define the Max Agent
    def maxAgent(gameState,depth,agent):
      
      if (depth==self.depth or gameState.isWin() or gameState.isLose()):
        return self.evaluationFunction(gameState)

      Result = ["nothing",-float("inf")]
      pacmanAction = gameState.getLegalActions(agent)

      if not pacmanAction:
        return self.evaluationFunction(gameState)

      for action in pacmanAction:
        currentgamestate = gameState.generateSuccessor(agent,action)
        nextResult = miniAgent(currentgamestate,depth,1)
        if type(nextResult) is list:
          testVal = nextResult[1]
        else:
          testVal = nextResult
        if testVal > Result[1]:
          Result = [action, testVal]
      return Result


    #define the Mini Agent
    def miniAgent(gameState,depth,agent):
      

      #next agent is maxAgent


      if agent >= (gameState.getNumAgents()-1):
        Result = ["nothing",float("inf")]
        GAction = gameState.getLegalActions(agent)
        if not GAction:
          return self.evaluationFunction(gameState)
        depth = depth + 1
        for action in GAction:
          currentgamestate = gameState.generateSuccessor(agent,action)
          nextResult = maxAgent(currentgamestate,depth,0)
          if type(nextResult) is list:
            testVal = nextResult[1]
          else:
            testVal = nextResult
          if testVal < Result[1]:
            Result = [action, testVal]
        return Result

      #next agent is MiniAgent
      else:
        Result = ["nothing",float("inf")]
        GAction = gameState.getLegalActions(agent)

        if not GAction:
          return self.evaluationFunction(gameState)

        for action in GAction:
          currentgamestate = gameState.generateSuccessor(agent,action)
          nextResult = miniAgent(currentgamestate,depth,agent+1)
          if type(nextResult) is list:
            testVal = nextResult[1]
          else:
            testVal = nextResult
          if testVal < Result[1]:
            Result = [action, testVal] 
        return Result


    Solution = maxAgent(gameState,0,0)
    return Solution[0]       



class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    Solution = []
    testVal = 0
    #define the Max Agent
    def maxAgent(gameState,depth,agent,alpha,beta):
      
      if (depth==self.depth or gameState.isWin() or gameState.isLose()):
        return self.evaluationFunction(gameState)

      Result = ["nothing",-float("inf")]
      pacmanAction = gameState.getLegalActions(agent)

      if not pacmanAction:
        return self.evaluationFunction(gameState)

      for action in pacmanAction:
        currentgamestate = gameState.generateSuccessor(agent,action)
        nextResult = miniAgent(currentgamestate,depth,1,alpha,beta)
        if type(nextResult) is list:
          testVal = nextResult[1]
        else:
          testVal = nextResult
        if testVal > Result[1]:
          Result = [action, testVal]
        alpha = max(alpha,testVal)
        if alpha > beta:
          return Result
      return Result


    #define the Mini Agent
    def miniAgent(gameState,depth,agent,alpha,beta):
      

      #next agent is maxAgent


      if agent >= (gameState.getNumAgents()-1):
        Result = ["nothing",float("inf")]
        GAction = gameState.getLegalActions(agent)
        if not GAction:
          return self.evaluationFunction(gameState)
        depth = depth + 1
        for action in GAction:
          currentgamestate = gameState.generateSuccessor(agent,action)
          nextResult = maxAgent(currentgamestate,depth,0,alpha,beta)
          if type(nextResult) is list:
            testVal = nextResult[1]
          else:
            testVal = nextResult
          if testVal < Result[1]:
            Result = [action, testVal]
          beta = min(beta,testVal)
          if alpha > beta:
            return Result
        return Result


      #next agent is MiniAgent
      else:
        Result = ["nothing",float("inf")]
        GAction = gameState.getLegalActions(agent)

        if not GAction:
          return self.evaluationFunction(gameState)

        for action in GAction:
          currentgamestate = gameState.generateSuccessor(agent,action)
          nextResult = miniAgent(currentgamestate,depth,agent+1,alpha,beta)
          if type(nextResult) is list:
            testVal = nextResult[1]
          else:
            testVal = nextResult
          if testVal < Result[1]:
            Result = [action, testVal]
          beta = min(beta,testVal)
          if alpha > beta:
            return Result 
        return Result


    Solution = maxAgent(gameState,0,0,-float("inf"),float("inf"))
    return Solution[0] 

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    "*** YOUR CODE HERE ***"
    Solution = []
    testVal = 0
    tempVal = 0
    #define the Max Agent
    def maxAgent(gameState,depth,agent):
      
      if (depth==self.depth or gameState.isWin() or gameState.isLose()):
        return self.evaluationFunction(gameState)

      Result = ["nothing",-float("inf")]
      pacmanAction = gameState.getLegalActions(agent)

      if not pacmanAction:
        return self.evaluationFunction(gameState)

      for action in pacmanAction:
        currentgamestate = gameState.generateSuccessor(agent,action)
        nextResult = expAgent(currentgamestate,depth,1)
        if type(nextResult) is list:
          testVal = nextResult[1]
        else:
          testVal = nextResult
        if testVal > Result[1]:
          Result = [action, testVal]
      return Result


    #define the Mini Agent
    def expAgent(gameState,depth,agent):
      

      #next agent is maxAgent


      if agent >= (gameState.getNumAgents()-1):
        Result = ["nothing",0]
        GAction = gameState.getLegalActions(agent)
        if not GAction:
          return self.evaluationFunction(gameState)
        probability = 1.0/len(GAction)
        depth = depth + 1
        for action in GAction:
          currentgamestate = gameState.generateSuccessor(agent,action)
          nextResult = maxAgent(currentgamestate,depth,0)
          if type(nextResult) is list:
            tempVal = nextResult[1]
          else:
            tempVal = nextResult
          Result[0] = action
          Result[1] += tempVal*probability
        return Result

      #next agent is MiniAgent
      else:
        Result = ["nothing",0]
        GAction = gameState.getLegalActions(agent)

        if not GAction:
          return self.evaluationFunction(gameState)
        probability = 1.0/len(GAction)
        for action in GAction:
          currentgamestate = gameState.generateSuccessor(agent,action)
          nextResult = expAgent(currentgamestate,depth,agent+1)
          if type(nextResult) is list:
            tempVal = nextResult[1]
          else:
            tempVal = nextResult
          Result[0] = action
          Result[1] += tempVal*probability 
        return Result


    Solution = maxAgent(gameState,0,0)
    return Solution[0]

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
  """
  "*** YOUR CODE HERE ***"
  score = 0
  PacPos = currentGameState.getPacmanPosition()
  Food = currentGameState.getFood()
  GhostStates = currentGameState.getGhostStates()
  cap = currentGameState.getCapsules()

  FoodList = Food.asList()
  FoodDistance = [0]
  GhostDistance = [0]
  GhostScore = [0]

  for food in FoodList:
    x = manhattanDistance(PacPos,food)
    FoodDistance.append(x)

  cloestFood = min(FoodDistance)

  GhostDangerTimer = 0
  for state in GhostStates:
      temp_d = manhattanDistance(PacPos, state.getPosition())
      GhostDistance.append(temp_d)
      if state.scaredTimer == 0:
        GhostDangerTimer ==100*(manhattanDistance(state.getPosition(), PacPos))
      else:
        GhostDangerTimer += 50

  GhostDistance.sort()
  GhostDangerZone = 0
  index_temp = 10
  for x in range (0,len(GhostDistance)-1):
    GhostDangerZone += GhostDistance[x]*index_temp
    index_temp = index_temp/2

  cloestGhost = GhostDistance[0]
  
  

  score = currentGameState.getScore() + 2*GhostDangerTimer + cloestGhost
  return score







# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()





score = 0
  PacPos = currentGameState.getPacmanPosition()
  Food = currentGameState.getFood()
  GhostStates = currentGameState.getGhostStates()
  cap = currentGameState.getCapsules()
  newScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]
  screatime = 0

  for i in newScaredTimes:
    screatime = screatime +1

  FoodList = Food.asList()
  FoodDistance = [0]
  GhostDistance = [0]

  for food in FoodList:
    x = manhattanDistance(PacPos,food)
    FoodDistance.append(x)

  cloestFood = min(FoodDistance)

  GhostDangerTimer = 0
  for state in GhostStates:
      temp_d = manhattanDistance(PacPos, state.getPosition())
      GhostDistance.append(temp_d)

  cloestGhost = GhostDistance[0]
  score = currentGameState.getScore() - cloestFood;

  if ( screatime > 0):
    return  score - 20*cloestGhost+screatime;     
  else:
    return  score + 20*cloestGhost;