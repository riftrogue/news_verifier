from langgraph import Graph, Node
from langgraph.graph_agents import *
from orchestrator import orchestration_layer
# Create nodes for each agent
domain_node = Node(domain_routing_agent)
actor_node = Node(prime_actor_agent)
source_node = Node(official_source_agent)
propaganda_node = Node(propaganda_agent)
rag_node = Node(rag_inconsistency_agent)

# Orchestration node
orchestration_node = Node(orchestration_layer)

# Create graph
graph = Graph()
graph.add_nodes([domain_node, actor_node, source_node, propaganda_node, rag_node, orchestration_node])

# Connect agent outputs to orchestration
for agent_node in [domain_node, actor_node, source_node, propaganda_node, rag_node]:
    graph.connect(agent_node, orchestration_node)

# Execute graph
news_text = "Breaking: Something sensational in Indian media."
outputs = graph.run(news_text=news_text)
print(outputs)
