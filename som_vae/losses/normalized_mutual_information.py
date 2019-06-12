def normalized_mutual_information(cluster_assignments, class_assignments):
    """Computes the Normalized Mutual Information between cluster and class assignments.
    Compare to https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html

    Args:
        cluster_assignments (list): List of cluster assignments for every point.
        class_assignments (list): List of class assignments for every point.

    Returns:
        float: The NMI value.
    """
    assert len(cluster_assignments) == len(class_assignments), "The inputs have to be of the same length."

    clusters = np.unique(cluster_assignments)
    classes = np.unique(class_assignments)

    num_samples = len(cluster_assignments)
    num_clusters = len(clusters)
    num_classes = len(classes)

    assert num_classes > 1, "There should be more than one class."

    cluster_class_counts = {cluster_: {class_: 0 for class_ in classes} for cluster_ in clusters}

    for cluster_, class_ in zip(cluster_assignments, class_assignments):
        cluster_class_counts[cluster_][class_] += 1

    cluster_sizes = {cluster_: sum(list(class_dict.values())) for cluster_, class_dict in cluster_class_counts.items()}
    class_sizes = {class_: sum([cluster_class_counts[clus][class_] for clus in clusters]) for class_ in classes}

    I_cluster_class = H_cluster = H_class = 0

    for cluster_ in clusters:
        for class_ in classes:
            if cluster_class_counts[cluster_][class_] == 0:
                pass
            else:
                I_cluster_class += (cluster_class_counts[cluster_][class_]/num_samples) * \
                (np.log((cluster_class_counts[cluster_][class_]*num_samples)/ \
                        (cluster_sizes[cluster_]*class_sizes[class_])))

    for cluster_ in clusters:
        H_cluster -= (cluster_sizes[cluster_]/num_samples) * np.log(cluster_sizes[cluster_]/num_samples)

    for class_ in classes:
        H_class -= (class_sizes[class_]/num_samples) * np.log(class_sizes[class_]/num_samples)

    NMI = (2*I_cluster_class)/(H_cluster+H_class)

    return NMI
