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


from cmath import inf
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
        #print("successorGameState : " , successorGameState)
        newPacmanPosition = successorGameState.getPacmanPosition()
        #print("newPos : " , newPacmanPosition)
        newFoodPosition = successorGameState.getFood()
        #print("newFood : " , newFoodPosition)
        newGhostStates = successorGameState.getGhostStates()
        #print("newGhost : " , newGhostStates)
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        #print("newScaredTimes : " , newScaredTimes)

        "*** YOUR CODE HERE ***"
        foodDistance = 0
        #minFoodDistance = []
        #pacmanPosition = list(newPacmanPosition)
        foodList = currentGameState.getFood().asList()

        for foodPosition in foodList:
            foodDistance = [manhattanDistance(newPacmanPosition,foodPosition)]
            #minFoodDistance.append(foodDistance)
            closestFood = min(foodDistance)
            #if foodDistance < minFoodDistance:
                #minFoodDistance.append(foodDistance)
                #minFoodDistance = foodDistance

        #minFoodDistance = -minFoodDistance

        for ghostState in newGhostStates:
            ghostDistance = [manhattanDistance(newPacmanPosition,ghostState.getPosition())]
            closestGhost = min(ghostDistance)
            if closestGhost < 2:
                return -200

            if action == "STOP":
                return -200      

        return (-closestFood)
  

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
        """Returns the minimax action from the current gameState using self.depth
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

        def maximum(gameState, currentDepth, agentIndex):
            currentDepth -= 1
            if currentDepth < 0 or gameState.isLose() or gameState.isWin():
                return (self.evaluationFunction(gameState),None)
            v = float("-inf")
    
            actionList = []
            nextAction = gameState.getLegalActions(agentIndex)
            #print(nextAction)
            for action in nextAction:
                #print("action in max : " , action)
                successorValue = gameState.generateSuccessor(agentIndex, action)
                maxActionsPossible = minimum(successorValue, currentDepth, agentIndex+1)[0]
                #print("actions possible in max: ", maxActionsPossible)
                actionList.append(maxActionsPossible)
                #print("action List max : " ,maxActionsPossible)
                
                maxScore = max(actionList)  
                    #print("vMax : ", maxScore)
                if maxScore > v:
                    v = maxScore
            
                maxIndices = [index for index in range(len(actionList)) if actionList[index] == maxScore]
                #print("maxIndices :" ,maxIndices)
                #print("vMax, maxAction : ", v, nextAction[random.choice(maxIndices)])
            return v, nextAction[random.choice(maxIndices)]

        def minimum(gameState, currentDepth, agentIndex):
            if currentDepth < 0 or gameState.isLose() or gameState.isWin():
                return (self.evaluationFunction(gameState),None)
            v = float("inf")

            actionList = []
            #minAction = action
            evalfunc, nextAgent = (minimum, agentIndex+1) if agentIndex < gameState.getNumAgents()-1 else (maximum, 0)
            nextAction = gameState.getLegalActions(agentIndex)
            #print(nextAction)
            for action in nextAction:
                #print("action in min : " , action)
                successorValue = gameState.generateSuccessor(agentIndex, action)
                minActionsPossible = evalfunc(successorValue, currentDepth, nextAgent)[0]
                #print("actions possible in min: ", minActionsPossible)
                actionList.append(minActionsPossible)
                #print("action List min: " ,minActionsPossible)
                minScore = min(actionList)
                if minScore < v:
                    v = minScore
            
                minIndices = [index for index in range(len(actionList)) if actionList[index] == minScore]
                #print("minIndices :" ,minIndices)
                #print("vMin, minAction : ", v, nextAction[random.choice(minIndices)])
            return v, nextAction[random.choice(minIndices)]
        return maximum(gameState, self.depth, 0)[1]       

        #util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
       
        def maximum(gameState, maxScore, minScore, currentDepth, agentIndex):
            currentDepth -= 1
            if currentDepth < 0 or gameState.isLose() or gameState.isWin():
                return (self.evaluationFunction(gameState),None)
            v = float("-inf")
            maxAction = []
            nextAction = gameState.getLegalActions(agentIndex)
            #print(nextAction)
            for action in nextAction:
                #print("action in max : " , action)
                successorValue = gameState.generateSuccessor(agentIndex, action)
                maxActionsPossible = minimum(successorValue, maxScore, minScore, currentDepth, agentIndex+1)[0]
                #print("actions possible in max: ", maxActionsPossible)
                #actionList.append(maxActionsPossible)
                #print("action List max : " ,maxActionsPossible) 
                
                if maxActionsPossible > v:
                    v = maxActionsPossible
                    maxAction = [action] 
                    #print("best action max : ", maxAction)
                elif maxActionsPossible == v:
                    maxAction.append(action)
                if v > minScore:
                    break

                maxScore = max(v, maxScore)      
                #print("vMax : ", maxScore)   
                #print("vMax, maxAction : ", v, random.choice(maxAction))
            return v, random.choice(maxAction)

        def minimum(gameState, maxScore, minScore, currentDepth, agentIndex):
            if currentDepth < 0 or gameState.isLose() or gameState.isWin():
                return (self.evaluationFunction(gameState),None)
            v = float("inf")
            
            minAction = []
            evalfunc, nextAgent = (minimum, agentIndex+1) if agentIndex < gameState.getNumAgents()-1 else (maximum, 0)
            nextAction = gameState.getLegalActions(agentIndex)
            #print(nextAction)
            for action in nextAction:
                #print("action in min : " , action)
                successorValue = gameState.generateSuccessor(agentIndex, action)
                minActionsPossible = evalfunc(successorValue, maxScore, minScore, currentDepth, nextAgent)[0]
                #print("actions possible in min: ", minActionsPossible)
                #actionList.append(minActionsPossible)
                #print("action List min: " ,minActionsPossible)
                minScore = min(v, minScore) 
                #print("vMin : ", minScore)
                if minActionsPossible < v:
                    v = minActionsPossible
                    minAction = [action]
                    print("best action min : ", minAction)
                elif minActionsPossible == v:
                    minAction.append(action)
                if maxScore > v:
                    break        
            
                minScore = min(v, minScore) 
                #print("vMin : ", minScore)
                #minScore = min(v, minScore)    
                #print("vMin, minAction : ", v, random.choice(minAction))
            return v, random.choice(minAction)
        return maximum(gameState, float("-inf"), float("inf"), self.depth, 0)[1] 
        util.raiseNotDefined()

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
        def maximum(gameState, currentDepth, agentIndex):
            currentDepth -= 1
            if currentDepth < 0 or gameState.isLose() or gameState.isWin():
                return (self.evaluationFunction(gameState),None)
            v = float("-inf")
            actionList = []
            nextAction = gameState.getLegalActions(agentIndex)
            #print(nextAction)
            for action in nextAction:
                #print("action in max : " , action)
                successorValue = gameState.generateSuccessor(agentIndex, action)
                maxActionsPossible = expected(successorValue, currentDepth, agentIndex+1)[0]
                #print("actions possible in max: ", maxActionsPossible)
                actionList.append(maxActionsPossible)
                #print("action List max : " ,maxActionsPossible)
                maxScore = max(actionList)  
                    #print("vMax : ", maxScore)
                if maxScore > v:
                    v = maxScore
            
                maxIndices = [index for index in range(len(actionList)) if actionList[index] == maxScore]
                #print("maxIndices :" ,maxIndices)
                #print("vMax, maxAction : ", v, nextAction[random.choice(maxIndices)])
            return v, nextAction[random.choice(maxIndices)]

        def expected(gameState, currentDepth, agentIndex):
            if currentDepth < 0 or gameState.isLose() or gameState.isWin():
                return (self.evaluationFunction(gameState),None)
            v = float("inf")
            actionList = []
            evalfunc, nextAgent = (expected, agentIndex+1) if agentIndex < gameState.getNumAgents()-1 else (maximum, 0)
            nextAction = gameState.getLegalActions(agentIndex)
            #print(nextAction)
            for action in nextAction:
                #print("action in min : " , action)
                successorValue = gameState.generateSuccessor(agentIndex, action)
                avgActionsPossible = evalfunc(successorValue, currentDepth, nextAgent)[0]
                #print("actions possible in min: ", minActionsPossible)
                actionList.append(avgActionsPossible)
                #print("action List min: " , avgActionsPossible)
                totalScore = sum(actionList)
                #print("total score : ", totalScore)
                avgScore = totalScore / len(actionList)
                #print("average score : ", avgScore)
            # print("minIndices :" ,minIndices)
            #print("vMin, avgAction : ", avgScore, action)
            return avgScore, action
        return maximum(gameState, self.depth, 0)[1]      
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    #successorGameState = currentGameState.generatePacmanSuccessor(action)
    #print("successorGameState : " , successorGameState)
    newPacmanPosition = currentGameState.getPacmanPosition()
    #print("newPos : " , newPacmanPosition)
    newFoodPosition = currentGameState.getFood()
    #print("newFood : " , newFoodPosition)
    newGhostStates = currentGameState.getGhostStates()
    #print("newGhost : " , newGhostStates)
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    #print("newScaredTimes : " , newScaredTimes)
    newPowerCapsulePosition = currentGameState.getCapsules()
    foodDistance = 0
    closestFood = 0
    closestPowerCapsule = 0
    foodList = currentGameState.getFood().asList()

    for foodPosition in foodList:
        foodDistance = [manhattanDistance(newPacmanPosition,foodPosition)]   
        closestFood = min(foodDistance)

    for powerCapsule in newPowerCapsulePosition:
        powerCapsuleDistance = [manhattanDistance(newPacmanPosition,powerCapsule)]
        closestPowerCapsule = min(powerCapsuleDistance)

    for ghostState in newGhostStates:
        ghostDistance = [manhattanDistance(newPacmanPosition,ghostState.getPosition())]
        closestGhost = min(ghostDistance)  

    return currentGameState.getScore()-(1*closestFood)-(3*closestGhost)-(1*closestPowerCapsule)
  
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
