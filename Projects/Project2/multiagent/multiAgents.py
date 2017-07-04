# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


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
        danger_threshold = 3 # lower than this, pacman will be scared of ghosts
        foodList = newFood.asList()
        if foodList:
            # get remaining food and ghosts
            foodLeft = successorGameState.getNumFood()
            ghostList = [state.getPosition() for state in newGhostStates]
            closeFoodDist, fpt = min([(manhattanDistance(newPos, p), p) for p in foodList])
            closeGhostDist, gpt = min([(manhattanDistance(newPos, p), p) for p in ghostList])
            eat_score = successorGameState.getScore()
            # check if ghost is near pacman
            if closeGhostDist < danger_threshold:
                # is a ghost is nearby, we want to return a low score
                # to simulate fear in pacman
                return 0
            # want food dist to have heavier impact on score, so have constant
            # of 10. For calculated dists above, we want to divide them by 1
            # and multiply the constants of weight/emphasis
            return 10*(1.0/closeFoodDist) + eat_score + (1.0/foodLeft) + 0.1*closeGhostDist
        return successorGameState.getScore()


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

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        max_depth = self.depth * gameState.getNumAgents()
        # uses value helper function below to get best action
        _, action = value(gameState, max_depth, self.index, self.evaluationFunction)
        return action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # initialize alpha and beta values
        alpha = float("-inf")
        beta = float("inf")
        max_depth = self.depth * gameState.getNumAgents()
        _, action = value(gameState, max_depth, self.index, self.evaluationFunction, alpha, beta)
        return action


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
        max_depth = self.depth * gameState.getNumAgents()
        # use helper function below, but set exp_val to True, so function uses
        # expected_value() function instead of getMin()
        _, action = value(gameState, max_depth, self.index, self.evaluationFunction, exp_val='True')
        return action


def value(state, depth, agent_idx, evaluate, alpha=None, beta=None, exp_val=False):
    if depth == 0 or state.isWin() or state.isLose():
        return evaluate(state), None
    if agent_idx == 0:
        # handle max
        return getMax(state, depth, agent_idx, evaluate, alpha, beta, exp_val)
    if agent_idx > 0:
        # handle min
        if exp_val:
            # for expectimax agent
            return expected_value(state, depth, agent_idx, evaluate)
        return getMin(state, depth, agent_idx, evaluate, alpha, beta, exp_val)


def getMax(state, depth, agent_idx, evaluate, alpha, beta, exp_val):
    best_max = float("-inf")
    best_action = None
    new_agent = (agent_idx+1) % state.getNumAgents()
    for action in state.getLegalActions(agent_idx):
        next_state = state.generateSuccessor(agent_idx, action)
        curr_max, act = value(next_state, depth-1, new_agent, evaluate, alpha, beta, exp_val)
        best_max = max(best_max, curr_max)
        # need to use alpha beta pruning check
        if alpha != None and beta != None:
            # for alpha beta agent
            if best_max > beta and best_max == curr_max:
                # can prune
                best_action = action
                return best_max, best_action
            alpha = max(alpha, best_max)
        if best_max == curr_max:
            best_action = action
    return best_max, best_action


def getMin(state, depth, agent_idx, evaluate, alpha, beta, exp_val):
    best_min = float("inf")
    best_action = None
    new_agent = (agent_idx+1) % state.getNumAgents()
    for action in state.getLegalActions(agent_idx):
        next_state = state.generateSuccessor(agent_idx, action)
        curr_min, act = value(next_state, depth-1, new_agent, evaluate, alpha, beta, exp_val)
        best_min = min(best_min, curr_min)
        # need to use alpha beta pruning check
        if alpha != None and beta != None:
            # for alpha beta agent
            if best_min < alpha and best_min == curr_min:
                # can prune
                best_action = action
                return best_min, best_action
            beta = min(beta, best_min)
        if best_min == curr_min:
            best_action = action
    return best_min, best_action


def expected_value(state, depth, agent_idx, evaluate):
    best_val = 0
    best_action = None
    new_agent = (agent_idx+1) % state.getNumAgents()
    num_actions = len(state.getLegalActions(agent_idx))
    for action in state.getLegalActions(agent_idx):
        next_state = state.generateSuccessor(agent_idx, action)
        prob = 1.0/num_actions
        curr_val, act = value(next_state, depth-1, new_agent, evaluate, exp_val='True')
        best_val += prob * curr_val
        best_action = action
    return best_val, best_action


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    "*** YOUR CODE HERE ***"
    foodList = newFood.asList()
    if foodList:
        foodLeft = currentGameState.getNumFood()
        ghostList = [state.getPosition() for state in newGhostStates]
        closeFoodDist, fpt = min([(manhattanDistance(newPos, p), p) for p in foodList])
        closeGhostDist, gpt = min([(manhattanDistance(newPos, p), p) for p in ghostList])
        eat_score = currentGameState.getScore()
        # account for ghost being near pacman
        if closeGhostDist < 3:
            if sum(newScaredTimes) == 0:
                return 0
            else:
                return 10*(1.0/closeFoodDist) + eat_score + (1.0/foodLeft) + 10*closeGhostDist
        return 10*(1.0/closeFoodDist) + eat_score + (1.0/foodLeft) + 0.1*closeGhostDist
    return currentGameState.getScore()

# Abbreviation
better = betterEvaluationFunction
