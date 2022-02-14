"""This is not a message passing layer yet"""


def list_of_neighboors(graph):
    lst = []
    for i in range(graph.num_nodes):
        node_list = sorted(graph.edge_index[0][1, :][(graph.edge_index[0][0, :] == i)].tolist())
        lst.append([[x] for x in node_list])
    return lst


def k_to_k_plus_one(graph, k_list):
    return_lst = []
    for i in range(graph.num_nodes):
        tmp_lst = []
        i_nbhd = graph.edge_index[0][1, :][(graph.edge_index[0][0, :] == i)].tolist()
        for lst in k_list[i]:
            for j in i_nbhd:
                if lst in k_list[j]:
                    k_1_clique = sorted(lst + [j])
                    if k_1_clique not in tmp_lst:
                        tmp_lst.append(k_1_clique)
        return_lst.append(tmp_lst)
    return return_lst
