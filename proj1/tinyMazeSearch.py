import util


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """

    class SearchNode:
        """
        Creates node: <state, action, parent_node>
        """

        def __init__(self, state, action=None, parent=None):
            self.state = state
            self.action = action
            self.parent = parent

        def extract_solution(self):
            """Gets complete path from goal state to parent node"""
            action_path = []
            search_node = self
            while search_node:
                if search_node.action:
                    action_path.append(search_node.action)
                search_node = search_node.parent
            return list(reversed(action_path))

    start_node = SearchNode(problem.getStartState())

    if problem.isGoalState(start_node.state):
        return start_node.extract_solution()

    frontier = util.Stack()
    explored = set()
    frontier.push(start_node)

    # run until stack is empty
    while not frontier.isEmpty():
        node = frontier.pop()  # choose the deepest node in frontier
        explored.add(node.state)

        if problem.isGoalState(node.state):
            return node.extract_solution()

        # expand node
        successors = problem.getSuccessors(node.state)

        for succ in successors:
            # make-child-node
            child_node = SearchNode(succ[0], succ[1], node)
            if child_node.state not in explored:
                frontier.push(child_node)

    # no solution
    util.raiseNotDefined()
