# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first
    [2nd Edition: p 75, 3rd Edition: p 87]

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm
    [2nd Edition: Fig. 3.18, 3rd Edition: Fig 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
            print "Start's successors:", problem.getSuccessors(node_location);

    """
    "*** YOUR CODE HERE ***"
    my_solution = [];
    visited_nodes = [];
    current_successor = [];
    my_node = util.Stack();

    my_node.push((problem.getStartState(),[]));
    while True:
        if my_node.isEmpty():
            return;
        node = my_node.pop();
        node_location = node[0];
        my_solution = node[1];
        if problem.isGoalState(node_location):
            return my_solution;
        else:
            visited_nodes.append(node_location);
            current_successor = problem.getSuccessors(node_location);
            for i in range(len(current_successor)):
                if current_successor[i][0] not in visited_nodes:
                    my_node.push((current_successor[i][0],my_solution + [current_successor[i][1]]));





    return my_solution;
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    [2nd Edition: p 73, 3rd Edition: p 82]
    """
    "*** YOUR CODE HERE ***"
    my_solution = [];
    visited_nodes = [];
    current_successor = [];
    my_node = util.Queue();

    my_node.push((problem.getStartState(),[]));
    while True:
        if my_node.isEmpty():
            return my_solution;
        node = my_node.pop();
        node_location = node[0];
        my_solution = node[1];
        if node_location not in visited_nodes:
            visited_nodes.append(node_location);
            current_successor = problem.getSuccessors(node_location);
            for i in range(len(current_successor)):
                if problem.isGoalState(current_successor[i][0]):
                    return my_solution + [current_successor[i][1]];
                if current_successor[i][0] not in visited_nodes:
                    my_node.push((current_successor[i][0],my_solution + [current_successor[i][1]]));
    util.raiseNotDefined()

def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    "*** YOUR CODE HERE ***"
    my_solution = [];
    visited_nodes = [];
    current_successor = [];
    my_node = util.PriorityQueue();

    my_node.push((problem.getStartState(),[]),0);
    while True:
        if my_node.isEmpty():
            return my_solution;
        node = my_node.pop();
        node_location = node[0];
        my_solution = node[1];
        if problem.isGoalState(node_location):
            return my_solution;
        visited_nodes.append(node_location);
        current_successor = problem.getSuccessors(node_location);
        for i in range(len(current_successor)):
            cost = problem.getCostOfActions(my_solution + [current_successor[i][1]]);
            if current_successor[i][0] not in visited_nodes:
                my_node.push((current_successor[i][0],my_solution + [current_successor[i][1]]),cost);
    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    "*** YOUR CODE HERE ***"
    my_solution = [];
    visited_nodes = [];
    current_successor = [];
    my_node = util.PriorityQueue();

    my_node.push((problem.getStartState(),[]),0);
    while True:
        if my_node.isEmpty():
            return my_solution;
        node = my_node.pop();
        node_location = node[0];
        my_solution = node[1];
        if problem.isGoalState(node_location):
            return my_solution;
        visited_nodes.append(node_location);
        current_successor = problem.getSuccessors(node_location);
        for i in range(len(current_successor)):
            h_n = heuristic(current_successor[i][0],problem);
            cost = problem.getCostOfActions(my_solution + [current_successor[i][1]])+h_n;
            if current_successor[i][0] not in visited_nodes:
                my_node.push((current_successor[i][0],my_solution + [current_successor[i][1]]),cost);
    util.raiseNotDefined()
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
