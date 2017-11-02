
__author__ = "Jeremie Lumbroso"

import networkx as _nx

##########################################################################

VI_PENDANT         = 'PENDANT'
VI_TRUE_TWIN       = 'TRUE_TWIN'
VI_FALSE_TWIN      = 'FALSE_TWIN'
VI_TRUE_ANTI_TWIN  = 'TRUE_ANTI-TWIN'
VI_FALSE_ANTI_TWIN = 'FALSE_ANTI-TWIN'
VI_C4              = 'C4'

##########################################################################

def is_subclique(g, nodes):
  """
  Check whether a subset of nodes given by ``nodes`` in graph ``g`` form
  an induced clique, that is whether the induced subgraph formed by the
  given nodes is a clique.

  :param networkx.classes.graph.Graph g: The graph to examine.
  
  :param nodes: The list of identifiers of the nodes to consider.
  
  :type nodes: List[int]
  
  :rtype: bool
  
  :returns: `True` if the nodes form an induced clique, `False` otherwise.
  """
  
  nodes_set = set(nodes)
  k = len(nodes_set)
  
  for u in nodes:
    neighbors = set(g.neighbors(u))
    induced_neighbors = neighbors.intersection(nodes_set)
    
    # Check whether the induced neighbors (neighbors of the current vertex
    # which are part of the given subset of nodes) is equal to k-1 (the size
    # of the subset of nodes, with the current node excluded).
    
    if len(induced_neighbors) != k-1:
      
      # If not, then the nodes do not form an induced clique.
      
      return False
  
  return True



##########################################################################

def __new_node_id(g):
  """
  Helper private function to return a new node identifier, which is one
  more than the largest existing identifier.

  :param networkx.classes.graph.Graph g: The graph for which a new unique
      identifier is desired.

  :rtype: int

  :returns: A unique integer that can be used as new node identifier
  """
  return max([-1] + g.nodes()) + 1

def __init_parameters(g, v, u = None, defensive = False):
  """
  Helper private method to initialize parameters of a vertex incremental
  operation. In particular: create a new identifier if necessary, and make
  a defensive copy of the graph `g` if requested.
  
  :param networkx.classes.graph.Graph g: The graph that will be modified.
  
  :param int v: An existing node of graph `g`.
  
  :param u: :param u: The (optional) identifier of the node that will be
      added to graph `g`.
  
  :type u: Any[int, None]
  
  :param bool defensive: Whether to make a defensive copy of graph `g`
      before extending it.
  
  :rtype: Tuple[networkx.classes.graph.Graph, int, int, bool]
  
  :returns: The tuple of original parameters, some of which may have
      been updated.
  """
  # If the identifier of the new node is omitted, generate one.
  if u == None:
    u = __new_node_id(g)
    
  # If defensive copying is requested, make a copy of the graph.
  if defensive:
    g = g.copy()
    
  return (g, v, u, defensive)



##########################################################################

def add_pendant(g, v, u = None, defensive = False):
  """
  Add a pendant node to graph `g`, attached to node `v`.
  
  :param networkx.classes.graph.Graph g: The graph to extend with a
      pendant node.
  
  :param int v: The existing node to which a pendant node must be
      attached.
  
  :param u: The (optional) identifier of the node that will be created as
      a pendant node to `v`.
  
  :type u: Any[int, None]
  
  :param bool defensive: Whether to make a defensive copy of graph `g`
      before extending it.
  
  :rtype: Tuple[networkx.classes.graph.Graph, int]
  
  :returns: A tuple of the graph `g` in which a pendant node has been
      attached to existing node `v`; and the identifier of the new node
      `u`.
  """
  
  # Initialize parameters.
  (g, v, u, defensive) = __init_parameters(g, v, u, defensive)
  
  # Add a pendant node to graph g.
  g.add_node(u)
  assert(u != v)
  g.add_edge(u, v)
  
  return (g, u)


def __add_twin(g, v, u = None,
               u_to_v = False, complement = False,
               defensive = False):
  """
  Helper private method to add a node to graph `g` according to one of the
  following vertex incremental operations:

  - true twin (if ``u_to_v`` is `True` and ``complement`` is `False`);

  - false twin (if ``u_to_v`` is `False` and ``complement`` is `False`);

  - true anti-twin (if ``u_to_v`` is `True` and ``complement`` is `True`);

  - false anti-twin (if ``u_to_v`` is `False` and ``complement`` is
    `True`).
  
  :param networkx.classes.graph.Graph g: The graph to extend with a new
      node.
  
  :param int v: The identifier of the existing node which will be extended
      by the vertex incremental operation.
  
  :param u: The (optional) identifier of the node that will be created as
      a vertex incremental extension to `v`.
  
  :type u: Any[int, None]
  
  :param bool u_to_v: Whether to connect existing node `u` to newly
      created twin node `v` (thereby either creating a true or false
      twin/anti-twin).

  :param bool complement: Whether to connect node `u` to the neighborhood
      of `v` (thereby creating a twin), or to the complement of the
      neighborhood of `v` (thereby creating an anti-twin).

  :param bool defensive: Whether to make a defensive copy of graph `g`
      before extending it.
  
  :rtype: Tuple[networkx.classes.graph.Graph, int]
  
  :returns: A tuple of the graph `g` which has been extended; and the
      identifier of the new node `u`.
  """
  
  # Initialize parameters.
  (g, v, u, defensive) = __init_parameters(g, v, u, defensive)
  
  # Create the new vertex
  assert(not u in g.nodes())
  g.add_node(u)
  
  # Connect u to the rest of the graph.
  if not complement:
    
    # Connect u to all neighbors of v.
    for w in g.neighbors(v):
      assert(u != w)
      g.add_edge(u, w)
      
  else:
    
    # Connect u to the complement of the neighbors of v (that is, all
    # nodes not connected to v).
    
    for w in g.nodes():
      
      # If node w is not connected to v, then connect to u.
      if not g.has_edge(w,v) and w != v and w != u:
        assert(u != w)
        g.add_edge(u, w)
        
  # If u_to_v is True, connect u to v (allows for true or false
  # twin/anti-twin).
  if u_to_v:
    assert(u != v)
    g.add_edge(u, v)
  
        
  return (g, u)


def add_true_twin(g, v, u = None, u_to_v = False, defensive = False):
  """
  Add a node to graph `g`, which is a true twin to node `v`: that is,
  connected to the neighborhood of `v` and to `v` itself.
  
  :param networkx.classes.graph.Graph g: The graph to extend with a twin.
  
  :param int v: The existing node for which a true twin must be created.
  
  :param u: The (optional) identifier of the node that will be created as
      a true twin to `v`.
  
  :type u: Any[int, None]
  
  :param bool defensive: Whether to make a defensive copy of graph `g`
      before extending it.
  
  :rtype: Tuple[networkx.classes.graph.Graph, int]
  
  :returns: A tuple of the graph `g` in which a true twin has been
      attached to existing node `v`; and the identifier of the new node
      `u`.
  """
  return __add_twin(g, v, u = u,
                    u_to_v = True, complement = False,
                    defensive = defensive)

def add_false_twin(g, v, u = None, u_to_v = False, defensive = False):
  """
  Add a node to graph `g`, which is a false twin to node `v`: that is,
  connected to the neighborhood of `v` and but not to `v` itself.
  
  :param networkx.classes.graph.Graph g: The graph to extend with a twin.
  
  :param int v: The existing node for which a false twin must be created.
  
  :param u: The (optional) identifier of the node that will be created as
      a false twin to `v`.
  
  :type u: Any[int, None]
  
  :param bool defensive: Whether to make a defensive copy of graph `g`
      before extending it.
  
  :rtype: Tuple[networkx.classes.graph.Graph, int]
  
  :returns: A tuple of the graph `g` in which a false twin has been
      attached to existing node `v`; and the identifier of the new node
      `u`.
  """
  return __add_twin(g, v, u = u,
                    u_to_v = False, complement = False,
                    defensive = defensive)

def add_true_anti_twin(g, v, u = None, u_to_v = False, defensive = False):
  """
  Add a node to graph `g`, which is a true anti-twin to node `v`: that
  is, connected to the complement of the neighborhood of `v` and to `v`
  itself.
  
  :param networkx.classes.graph.Graph g: The graph to extend with a true
      anti-twin.
  
  :param int v: The existing node for which a true anti-twin must be
      created.
  
  :param u: The (optional) identifier of the node that will be created as
      a true anti-twin to `v`.
  
  :type u: Any[int, None]
  
  :param bool defensive: Whether to make a defensive copy of graph `g`
      before extending it.
  
  :rtype: Tuple[networkx.classes.graph.Graph, int]
  
  :returns: A tuple of the graph `g` in which a true anti-twin has been
      attached to existing node `v`; and the identifier of the new node
      `u`.
  """
  return __add_twin(g, v, u = u,
                    u_to_v = True, complement = True,
                    defensive = defensive)

def add_false_anti_twin(g, v, u = None, u_to_v = False, defensive = False):
  """
  Add a node to graph `g`, which is a false anti-twin to node `v`: that
  is, connected to the complement of the neighborhood of `v` and to `v`
  itself.
  
  :param networkx.classes.graph.Graph g: The graph to extend with a false
      anti-twin.
  
  :param int v: The existing node for which a false anti-twin must be
      created.
  
  :param u: The (optional) identifier of the node that will be created as
      a false anti-twin to `v`.
  
  :type u: Any[int, None]
  
  :param bool defensive: Whether to make a defensive copy of graph `g`
      before extending it.
  
  :rtype: Tuple[networkx.classes.graph.Graph, int]
  
  :returns: A tuple of the graph `g` in which a false anti-twin has been
      attached to existing node `v`; and the identifier of the new node
      `u`.
  """
  return __add_twin(g, v, u = u,
                    u_to_v = False, complement = False,
                    defensive = defensive)

def add_c4(g, v, a = None, b = None, c = None,
           u_to_v = False, defensive = False):
  """
  Add a three nodes to graph `g`, forming a cycle of size 4 wtih node `v`.
  
  :param networkx.classes.graph.Graph g: The graph to extend with a C_4.
  
  :param int v: The existing node with which a C_4 must be formed.s
  
  :param a: The (optional) identifier of the first node of the new cycle
      that will be created and attached to `v`.

  :param b: The (optional) identifier of the second node of the new cycle
      that will be created and attached to `a`.

  :param c: The (optional) identifier of the third node of the new cycle
      that will be created and attached to `b` and 'v'.
  
  :type a: Any[int, None]

  :type b: Any[int, None]

  :type c: Any[int, None]
  
  :param bool defensive: Whether to make a defensive copy of graph `g`
      before extending it.
  
  :rtype: Tuple[networkx.classes.graph.Graph, int]
  
  :returns: A tuple of the graph `g` in which a cycle of size 4 has been
      attached to existing node `v`; and the identifier of one of the
      new nodes 'a'
  """
  
  # Initialize parameters.
  (g, v, a, defensive) = __init_parameters(g, v, a, defensive)

  # Add a pendant node 'a' to 'v' in graph 'g'
  (g, a) = add_pendant(g, v, a, defensive)

  # Add a pendant node 'b' to 'a' in graph 'g'
  (g, b) = add_pendant(g, a, b, defensive)
  
  # Add a pendant node 'c' to 'b' in graph 'g'
  (g, c) = add_pendant(g, b, c, defensive)

  # connect 'c' to 'v' to complete the cycle
  assert(c != v)
  g.add_edge(c, v)

  return (g, a)


##########################################################################

def extend_all_nodes(g, operation, u = None,
                     no_isomorphic = True,
                     constraint = None, postprocess = None):
  """
  Extend an initial graph `g` in all possible ways, by independently
  applying a vertex incremental operation to each of the nodes of `g`,
  possibly subject to a constraint. It is possible to only obtain
  non-isomorphic graphs.

  :param networkx.classes.graph.Graph g: The graph to extend with a new
      node.
    
  :param operation: The vertex incremental operation by which to extend
      graph `g`.
  
  :type operation: Callable[[networkx.classes.graph.Graph, int,
                             Any[int, None], bool],
                            Tuple[networkx.classes.graph.Graph, int]]

  :param u: The (optional) identifier of the node that will be created as
      a vertex incremental extension.
  
  :type u: Any[int, None]

  :param bool no_isomorphic: Whether to filter out isomorphic graphs or
      not.

  :param constraint: A constraint function that will be given the original
      graph `g` as well as the node `v` being extended, and must return
      `True` if the vertex incremental operation can be applied, and
      `False` otherwise.
  
  :type constraint: Callable[[networkx.classes.graph.Graph, int], bool]

  :param constraint: A constraint function that will be given the original
      graph `g` as well as the node `v` being extended, and must return
      `True` if the vertex incremental operation can be applied, and
      `False` otherwise.
  
  :param postprocess: A function that optionally can write state
      information to each newly created graph (this callback function is
      provided the original graph, the new graph, the identifier of the
      node which was extended, and the the identifier of the newly created
      node).

  :type postprocess: Callable[[networkx.classes.graph.Graph,
                               networkx.classes.graph.Graph, int, int],
                              Any[networkx.classes.graph.Graph, None]]
  
  :rtype: List[networkx.classes.graph.Graph]
  
  :returns: A list of all newly produced graphs.
  """

  # The generated graphs need to be stored in a list to check for
  # isomorphic duplicates.
  
  new_graphs = []
  
  for v in g.nodes():
    
    if constraint != None and not constraint(g, v):
      # The constraint has return False, which means we should skip
      # applying the vertex incremental operation to this vertex v.
      continue
      
    (new_g, new_u) = operation(g, v, u, defensive = True)
    
    if no_isomorphic:
      is_duplicate = False
      
      # Check whether graph is isomorphic another one in the list.
      for other_g in new_graphs:
        if _nx.is_isomorphic(new_g, other_g):
          is_duplicate = True
          break
        
      # If graph is isomorphic to an existing one, ignore it.
      if is_duplicate:
        continue

    if postprocess != None:

      # Call the postprocessing method, and if it returns a graph then
      # substitute to the existing one.
      
      new_gp = postprocess(g, new_g, v, new_u)
      if new_gp != None:
        new_g = new_gp
      
    new_graphs.append(new_g)
  
  return new_graphs



##########################################################################

def only_non_isomorphic(graphs):
  """
  Filter a list of graphs by keeping only one graph per equivalence
  class, with respect to graph isomorphism.
  
  :param graphs: A list of graphs possibly containing graphs isomorphic to
      one another.
  
  :type graphs: List[networkx.classes.graph.Graph]
  
  :rtype: List[networkx.classes.graph.Graph]
  
  :returns: The list of graphs given as parameter of the function, from
      which all isomorphic graphs have filtered out.
  """
  new_l = []
  not_l = [] # we optionally keep track of graphs we remove (for debug)

  n = len(graphs)
  
  for i in range(n):
    
    keep_i = True
    
    for j in range(i+1, n):
      
      if _nx.is_isomorphic(graphs[i], graphs[j]):
        keep_i = False
        break
      
    if keep_i:
      new_l += [ graphs[i] ]
    else:
      not_l += [ graphs[i] ]
      
  return new_l



##########################################################################

__VI_OPERATIONS = {
  VI_PENDANT         : add_pendant,
  VI_TRUE_TWIN       : add_true_twin,
  VI_FALSE_TWIN      : add_false_twin,
  VI_TRUE_ANTI_TWIN  : add_true_anti_twin,
  VI_FALSE_ANTI_TWIN : add_false_anti_twin,
  VI_C4              : add_c4
}

def _get_operation_from_name(opname):
  return __VI_OPERATIONS[opname]



##########################################################################

class VertexIncrementalGenerator(object):
  
  @classmethod
  def generate(cls, size):
    """
    Generate all graphs of the specified size.
    
    :param int size: Size of the graphs to be generated.
    
    :rtype: List[networkx.classes.graph.Graph]
    
    :returns: List of graphs of the specified size.
    """
    return list(cls(size = size))
  
  @classmethod
  def enumerate(cls, size):
    """
    Return the number of graphs of the specified size.
    
    :param int size: Size of the graphs to be counted.
    
    :rtype: int
    
    :returns: Number of graphs of the specified size.
    """
    return len(list(cls(size = size)))
  
  @classmethod
  def _postprocess_factory(cls, operation):
    """
    Protected method to create, for a given operation `operation`, a
    default postprocessor that simply updates the `vi_sequence` metadata
    of a graph to document the sequence of vertex incremental operations
    by which it has been created.
    
    :param str operation: The name of the operation.

    :returns: A default postprocessor that updates the `vi_sequence`
        attribute of a newly created graph.
    """
    
    # The template function
    def _postprocess_template(old_g, new_g, v, u):
      try:
        vi_sequence = old_g.vi_sequence[:]
      except AttributeError:
        vi_sequence = []
      
      vi_sequence.append((operation, v, u))
      new_g.vi_sequence = vi_sequence
      
      return new_g
    
    return _postprocess_template
  
  @classmethod
  def _complete_defaults(cls, operations):
    """
    Protected method to extend a set of allowable vertex incremental
    operations with the default constraint and postprocessing methods,
    when none is defined.

    :param operations: The specification of allowable vertex incremental
        operations.

    :returns: The specification extended with all the default constraint
        and postprocessing methods.
    """
    
    always = (lambda g, node: True)
    
    # Optionally, unconstrained operations can be specified as a list
    # instead of a dictionary.
    
    if type(operations) is list:
      operations = dict(map(lambda opname: (opname, {}), operations))
      
    # Check existence of 'constraint' and 'postprocess' fields, and create
    # them if necessary.
    
    for opname in operations.keys():
      opdef = operations[opname]
      
      # Default constraint (none at all).
      if not 'constraint' in opdef:
        opdef['constraint'] = always
        
      # Default postprocessor.
      if not 'postprocess' in opdef:
        opdef['postprocess'] = cls._postprocess_factory(opname)
        
      operations[opname] = opdef
      
    return operations
      
  _operations = None
  
  def __init__(self, size=None, min_size = 2, max_size = 10, initial=None):
    self._operations = self._complete_defaults(self._operations)
    
    if size != None:
      if size == 0:
        raise Exception("Must provide non-zero size.")
      self.min_size = size
      self.max_size = size
    else:
      self.min_size = max(0, min_size)
      self.max_size = max_size
    
    self._steps = 0
    self._next_iter_id = -1
    
    if initial != None:
      self._initial = initial
    else:
      # By default, the initial graph is a clique of size 2 (K2).
      if self.min_size == 1:
        self._initial = _nx.complete_graph(1)
        self._next_iter_id = 0
      elif self.min_size >= 2:
        self._initial = _nx.complete_graph(2)
        if self.min_size == 2:
          self._next_iter_id = 0
    
    self._previous = [ self._initial ]
          
  def __iter__(self):
    """
    Return this iterator class itself.
    
    :rtype: VertexIncrementalGenerator
    
    :return: This iterator class itself.
    """
    return self
  
  def _next_graph(self):
    """
    Return the next item in the exhaustive sequences of graphs, ignoring
    specified size constraints.
    
    :rtype: networkx.classes.graph.Graph
    
    :returns: The next item in the exhaustive sequence of graphs
        generated by the allowed vertex incremental operations, and
        within the requested size constraints.
    
    :raises: StopIteration
    """
    # Determine if we need to build the next batch of graphs.
    if self._next_iter_id < 0 or self._next_iter_id >= len(self._previous):
      
      if len(self._previous) > 0:
        # Check whether the current batch (that has been exhausted) is of
        # maximal size; if that is the case, stop iteration.
        if len(self._previous[0].nodes()) == self.max_size:
          raise StopIteration("Max size graphs")
        
      self._next_batch_of_graphs()
      self._next_iter_id = 0
      
      # Check for abnormal error.
      if len(self._previous) == 0:
        raise StopIteration("Error when generating next batch of graphs.")
    
    k = self._next_iter_id
    self._next_iter_id += 1
    next_graph = self._previous[k]
    
    return next_graph
  
  def __next__(self):
    """
    Return the next item in the exhaustive sequences of graphs, subject to
    specified size constraints.
    
    :rtype: networkx.classes.graph.Graph
    
    :returns: The next item in the exhaustive sequence of graphs
        generated by the allowed vertex incremental operations, and
        within the requested size constraints.
    
    :raises: StopIteration
    """
    
    # Repeatedly draw graphs until they are of at least min_size.
    while True:
      next_graph = self._next_graph()
      if len(next_graph.nodes()) >= self.min_size:
        break
      
    # Shouldn't happen, but testing this anyway.
    if len(next_graph.nodes()) > self.max_size:
      raise StopIteration("Encountered graph beyond max size")
    
    return next_graph
  
  def next(self):
    """
    Return the next item in the exhaustive sequences of graphs, subject to
    specified size constraints.
    
    :rtype: networkx.classes.graph.Graph
    
    :returns: The next item in the exhaustive sequence of graphs
        generated by the allowed vertex incremental operations, and
        within the requested size constraints.
    
    :raises: StopIteration
    """
    return self.__next__()
    
  def _next_batch_of_graphs(self):
    """
    Generate a new set of graphs obtained from applying the allowed vertex
    incremental operations every possible way to the graphs of the
    previously generated (or initial) set of graphs.
    
    :rtype: List[networkx.classes.graph.Graph]
    
    :returns: The next batch of graphs generated by applying the allowed
        vertex incremental operations in every possible way, to the
        previous batch of graphs.
    """
    current = []
    
    for g in self._previous:
      
      for opname in self._operations:
        # Retrieve information on the operation.
        operation   = _get_operation_from_name(opname)
        constraint  = self._operations[opname]['constraint']
        postprocess = self._operations[opname]['postprocess']
        
        # Extend the current graph in all possible ways.
        new_graphs  = extend_all_nodes(g, operation,
                                       no_isomorphic = True,
                                       constraint = constraint,
                                       postprocess = postprocess)
        
        # Add the new graphs to the current list.
        current += new_graphs
        
    # Filter out isomorphic duplicates.
    current = only_non_isomorphic(current)
    
    # Update the state of this generator.
    self._previous = current[:]
    self._steps += 1
    
    return current



##########################################################################

class DistanceHereditaryGenerator(VertexIncrementalGenerator):
  
  def __init__(self, size):
    self._operations = [ VI_PENDANT, VI_TRUE_TWIN, VI_FALSE_TWIN ]
    super(DistanceHereditaryGenerator, self).__init__(size = size,                                                      
                                                      initial = None)
  
class ThreeLeafPowerGenerator(VertexIncrementalGenerator):
  
  @classmethod
  def _pendant_constraint(cls, g, v):
    try:
      vi_sequence = g.vi_sequence
    except AttributeError:
      # No prior history, so no constraint.
      return True
    
    last_vi_op = vi_sequence[-1]
    
    # Only allow a pendant operation if the previous (and thus all
    # preceding operations) are pendant operations.
    
    return (last_vi_op[0] == VI_PENDANT)
  
  def __init__(self, size):
    self._operations = {
      VI_PENDANT : {
        'constraint' : ThreeLeafPowerGenerator._pendant_constraint },
      VI_TRUE_TWIN  : {},
    }
    super(ThreeLeafPowerGenerator, self).__init__(size = size,
                                                  initial = None)

class CographGenerator(VertexIncrementalGenerator):
  
  def __init__(self, size):
    self._operations = [ VI_TRUE_TWIN, VI_FALSE_TWIN ]
    super(CographGenerator, self).__init__(size = size,
                                           initial = None)

class SwitchCographGenerator(VertexIncrementalGenerator):
  
  def __init__(self, size):
    self._operations = [ VI_TRUE_TWIN, VI_FALSE_TWIN,
                         VI_TRUE_ANTI_TWIN, VI_FALSE_ANTI_TWIN ]
    super(SwitchCographGenerator, self).__init__(size = size,
                                                 initial = None)

class SixTwoChordalBipartiteGenerator(VertexIncrementalGenerator):
  
  def __init__(self, size):
    self._operations = [ VI_PENDANT, VI_FALSE_TWIN ]
    super(SixTwoChordalBipartiteGenerator, self).__init__(size = size,
                                                          initial = None)

class PtolemaicGenerator(VertexIncrementalGenerator):
  
  @classmethod
  def _false_twin_constraint(cls, g, v):
    # Only allow a false twin if the neighbors of v form a clique.
    return is_subclique(g, g.neighbors(v))
  
  def __init__(self, size):
    self._operations = {
      VI_PENDANT    : {},
      VI_TRUE_TWIN  : {},
      VI_FALSE_TWIN : {
        'constraint' : PtolemaicGenerator._false_twin_constraint }
    }
    super(PtolemaicGenerator, self).__init__(size = size,
                                             initial = None)

class FourCactusGenerator(VertexIncrementalGenerator):
  
  def __init__(self, size):
    initial = _nx.complete_graph(1)

    self._operations = [ VI_C4 ]
    super(FourCactusGenerator, self).__init__(size = size, initial = initial)

  """  def __init__(self, size=None, min_size = 2, max_size = 10, initial=None):
    self._operations = self._complete_defaults(self._operations)
  """

##########################################################################
  
# Legacy methods

def pendants(g, u = None):
  return extend_all_nodes(g, operation = add_pendant, u = u)

def true_twins(g, u = None):
  return extend_all_nodes(g, operation = add_true_twin, u = u)

def false_twins(g, u = None):
  return extend_all_nodes(g, operation = add_false_twin, u = u)

def true_anti_twins(g, u = None):
  return extend_all_nodes(g, operation = add_true_anti_twin, u = u)

def false_anti_twins(g, u = None):
  return extend_all_nodes(g, operation = add_false_anti_twin, u = u)

# Misc methods

def draw_graph(graph):
  import networkx as nx
  import matplotlib.pyplot as plt
  
  plt.clf()
  nx.draw(graph)
  plt.show()

def draw_graph_alt(graph):
  import random
  import networkx as nx
  import matplotlib.pyplot as plt
  
  plt.clf()
  plt.figure(1,figsize=(8,8))
  pos = nx.graphviz_layout(graph, prog="neato")
  c = [ random.random() ] * nx.number_of_nodes(graph)
  nx.draw(graph, pos,
          node_size=40,
             node_color=c,
             vmin=0.0,
             vmax=1.0,
             with_labels=False
     )        
  plt.show()

def save_graph(graph, filename):
  import networkx as nx
  import matplotlib.pyplot as plt

  plt.clf()
  nx.draw(graph)
  plt.savefig(filename)
  plt.clf()

def draw_all_tlp(n, also_iso=True):
  gsp = [ x for x in ThreeLeafPowerGenerator(n) ]
  i = 1
  for g in gs:
    save_graph(g, "graphs/3lp/3lp_n%d_a_%02d.png" % (n,i))
    i += 1

def draw_all_dh(n, also_iso=True):
  gsp = [ x for x in DistanceHereditaryGenerator(n) ]
  i = 1
  for g in gs:
    save_graph(g, "graphs/dh/dh_n%d_a_%02d.png" % (n,i))
    i += 1


################################## Block Graphs ##################################

def is_weakly_geodetic(g):
  """
  Checks to see whether its graph argument is weakly geodetic.
  A graph is weakly geodetic if for every pair of vertices of distance
  2 there is a unique common neighbour of them.

  :param g: the graph
  :returns: whether g is weakly-geodetic
  :rtype:   bool
  """
  import networkx as nx
  
  all_paths = nx.all_pairs_shortest_path_length(g)
  for u in g.nodes():
      
    two_away = [node for node, dist in all_paths[u].items() if dist == 2]
    for v in two_away:
      common_neighbor_seen = False
      for node in g.nodes():
        if node in g.neighbors(u) and node in g.neighbors(v):
          if common_neighbor_seen:
            return False
          else:
            common_neighbor_seen = True
  return True


def draw_all_block(n, also_iso=True):
  import networkx as nx

  graphs = [ x for x in DistanceHereditaryGenerator(n)
            if is_weakly_geodetic(x) and nx.is_chordal(x) ]
  i = 1
  for g in graphs:
    save_graph(g, "graphs/block/block_n%d_a_%02d.png" % (n,i))
    i += 1


# used to compare manually with block graphs to confirm correctness
def draw_all_ptol(n, also_iso=True):
  import networkx as nx

  graphs = [ x for x in DistanceHereditaryGenerator(n)
            if nx.is_chordal(x) ]
  i = 1
  for g in graphs:
    save_graph(g, "graphs/ptol/ptol_n%d_a_%02d.png" % (n,i))
    i += 1


def draw_all_four_cactus(n, also_iso=True):
  graphs = FourCactusGenerator(n)
  i = 1
  for g in graphs:
    save_graph(g, "graphs/4cactus/4cactus_n%d_a_%02d.png" % (n,i))
    i += 1

# unlabeled enumeration
# 1, 1, 2, 4, 9, 22, 59, 165, ...

def block_enum(n):
  import networkx as nx
  return [ len([x for x in DistanceHereditaryGenerator(n)
                if is_weakly_geodetic(x) and _nx.is_chordal(x)])
           for n in range(1,n+1) ]

def four_cactus_enum(n):
  return [ len([x for x in FourCactusGenerator(n)])
           for n in range(1,n+1) ]

# labeled enumeration

class Wrapper(object):
  """ A Wrapper class used for maintaining a set data structure of adjacency matrices

  In order to generate labeled graphs from unlabeled ones, one can consider all
  possible labellings of the vertices and check if they lead to a new adjacency
  matrix. To do so, we need to maintain a dynamic data stucture that maintains the
  adjacency matrices that have already been seen, and can efficiently add a matrix
  if it is not already included. Python built-in sets can be used for this purpose.

  Adjancecy matrices given by networkx in the form of numpy matrices are used.
  However, these matrices are not comparable. This wrapper class provides comparison
  methods that make it possible for the set data structure to perform efficiently.
  """
  def __init__(self, mat):
    self.mat = mat

  def __cmp__(self, other):
    if self.mat.shape != other.mat.shape:
      raise Exception("Matrices don't have the same shape.")
    rows, cols = self.mat.shape
    for i in range(rows):
      for j in range(cols):
        ret = self.mat[i,j] - other.mat[i,j]
        if(ret != 0):
          return ret
    return 0

  def __eq__(self, other):
    return self.__cmp__(other) == 0
  def __ne__(self, other):
    return self.__cmp__(other) != 0
  def __lt__(self, other):
    return self.__cmp__(other) < 0
  def __le__(self, other):
    return self.__cmp__(other) <= 0
  def __gt__(self, other):
    return self.__cmp__(other) > 0
  def __ge__(self, other):
    return self.__cmp__(other) >= 0

  def __hash__(self):
    import numpy as np
    arr = np.squeeze(np.asarray(self.mat))
    return hash(tuple(map(tuple, arr)))


def label_graph(g):
  """
  Generates all distinct labeled graphs from an unlabeled graph

  :param g: an unlabeled graph
  :returns: all distinct graphs obtained from labelling g up to isomorphism
  :rtype:   list
  """
  from itertools import permutations
  import networkx as nx
  import numpy as np

  L = set()
  sz = len(g)
  M = nx.to_numpy_matrix(g)
  M2 = np.zeros((sz,sz))
      
  for p in permutations(range(sz)):        
    for i in range(sz):
      for j in range(sz):
        M2[i,j] = M[p[i],p[j]]
    if Wrapper(M2) not in L:
      L.add(Wrapper(M2))

  return list(L)

# 1, 4, 29, 311, 4447, 79745, ...

def labeled_block_enum(n):
  """ 
  Enumerates labeled block graphs smaller than a certain size up to isomorphism

  :param n: strict upper bound on the size of enumerated graphs
  :returns: a list representing the 1...n-1 terms of the counting sequence
  :rtype:   list
  """
  from itertools import permutations
  import networkx as nx
  import numpy as np

  ret = []

  for sz in range(2, n + 1):
    graphs = [ x for x in DistanceHereditaryGenerator(sz)
            if is_weakly_geodetic(x) and nx.is_chordal(x) ]
    count = 0
    for g in graphs:
      count += len(label_graph(g))
    ret.append(count)

  return ret


def labeled_four_cactus_enum(n):
  """ 
  Enumerates labeled four cacti smaller than a certain size up to isomorphism

  :param n: strict upper bound on the size of enumerated graphs
  :returns: a list representing the 1...n-1 terms of the counting sequence
  :rtype:   list
  """
  from itertools import permutations
  import networkx as nx
  import numpy as np

  ret = []

  for sz in range(2, n + 1):
    graphs = FourCactusGenerator(sz)
    count = 0
    for g in graphs:
      count += len(label_graph(g))
    ret.append(count)

  return ret


##########################################################################

import click

@click.command()
@click.option('--labeled/--unlabeled', default = False,
              help = 'Whether to enumerate labeled graphs.')
@click.option('--draw', is_flag = True, default = False)
@click.option('--size', prompt = 'Maximum size',
              help = 'Number of vertices in the graph.')
def run(labeled, draw, size):
  """ Enumerates labeled and unlabeled block graphs """
  """
  if draw:
    draw_all_block(int(size))
  if labeled:
    click.echo(labeled_block_enum(int(size)))
  else:
    click.echo(block_enum(int(size)))
  """
  if draw:
    draw_all_four_cactus(int(size))
  if labeled:
    click.echo(labeled_four_cactus_enum(int(size)))
  else:
    click.echo(four_cactus_enum(int(size)))

if __name__ == '__main__':
  run()
