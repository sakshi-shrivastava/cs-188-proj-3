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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        distance = float("inf")
        if successorGameState.isWin():
            return distance

      
        newFoodNext = newFood.asList()
        currentNumFood = currentGameState.getNumFood()
        successorNumFood = successorGameState.getNumFood()
        successorScore = successorGameState.getScore()
        distance = float("-Inf")
        #succsessor = successorScore - 2
        #newTime = min(newScaredTimes)

        for state in newGhostStates:
            getDistance = state.getPosition()
            GhostmanhattanDis = manhattanDistance(getDistance, newPos)
            if GhostmanhattanDis is 0.5:
                return distance

        
        food_Distance = min(manhattanDistance(newPos, food) for food in (newFoodNext))

        successor_f = 0
        if (currentNumFood > successorNumFood):
            successor_f = 0
            successor_f = 90

        reflex = successorScore - 2 * food_Distance + successor_f

        return reflex
       
       
def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agent Index):
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

       
           
          

        
        def maxValue(gameState,depth, agent):
          legalMov = gameState.getLegalActions(agent)
          
          if gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState), Directions.STOP
          if depth < 0 :
            return self.evaluationFunction(gameState), Directions.STOP  
          if depth == self.depth*gameState.getNumAgents() :
            return (None,self.evaluationFunction(gameState)) 
          

          v = float("-inf")

          for action in legalMov:
            newState = gameState.generateSuccessor(agent,action)
            score = minValue(newState,depth,agent+1)[0]
    
            if score > v:
              v = score
              maxAction = action
          return (v,maxAction)
        




        def minValue(gameState,depth,agent):
          legalAction = gameState.getLegalActions(agent)
          if gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState), Directions.STOP
          if depth < 0 :
            return self.evaluationFunction(gameState), Directions.STOP  
          if depth == self.depth*gameState.getNumAgents() :
            return (None,self.evaluationFunction(gameState)) 
          
          
          v = float("inf")
         
         # if agent < gameState.getNumAgents()-1
       

          for action in legalAction:
            successor = gameState.generateSuccessor(agent,action)
            score = maxValue(successor,depth,agent)[0]

            if score < v:
              v = score
              minAction = action          
          return (v,minAction)
        

        #return maxValue(gameState,self.depth,0)[1]
       
        



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        return self.getExpectedValue(gameState,0, self.depth)[1]

       
    def getExpectedValue(self, game_state,agent_index, depth):
        
        
        if game_state.isWin():
            return [self.evaluationFunction(game_state), None]
        if game_state.isLose():
            return [self.evaluationFunction(game_state), None]
        if depth == 0:
            return [self.evaluationFunction(game_state), None]
        if game_state.getLegalActions(agent_index) == 0:
           return [self.evaluationFunction(game_state), None]
           
       
        if agent_index is not 0:
            move_legal1 = game_state.getLegalActions(agent_index)
            val= agent_index
            ideal_action1 = Directions.STOP
            newIndex = agent_index + 1*1
            prob = 1.0 * float(len(move_legal1))
            #flo = float("inf")
            numAgents = game_state.getNumAgents()
            index1 = (agent_index + 1) % numAgents
            if newIndex == numAgents:
                depth = depth - 1
            for act in move_legal1:
                successorState = game_state.generateSuccessor(agent_index, act)
                value_expecti, _ = self.getExpectedValue(successorState, index1, depth)
                val = val + value_expecti/ prob
            return val, ideal_action1
            
           

        if agent_index is 0:
            Value_Max = float("-inf")
            value = Value_Max
            ideal_action2 = Directions.STOP
            move_legal2 = game_state.getLegalActions(0)
            index2 = (agent_index + 1) % game_state.getNumAgents()
            for act in move_legal2:
                successorState = game_state.generateSuccessor(0, act)
                pac_score, _ = self.getExpectedValue(successorState, index2, depth)
                if max(value, pac_score) == pac_score:
                   value, ideal_action2 = pac_score, act
            return (value, ideal_action2)
        
        #   util.raiseNotDefined()
        

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPosition = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhost = currentGameState.getGhostStates()
    gameScore = currentGameState.getScore()    #current score of succesor state

    if currentGameState.isWin():
        return float("inf")
    if currentGameState.isLose():
        return float("-inf")
    
    #for each ghost find the distance from pacman 
    # for ghost in newGhost:
    #    if manhattanDistance(newPosition, ghost.getPosition()) > 0:

    # Thought: If the ghost can be eaten, and the ghost is near, and the distance is small.
    # In order to get a bigger score we divide the distance to a big number to get a higher score
    #  If the ghost cannot be eaten, and the ghost is far, and the distance is big. We want to avoid 
    # such situation so we subtract the distance to a big number to lower the score and avoid this state.
          

# Abbreviation
better = betterEvaluationFunction
