import numpy as np


def cosine_similarity(point, centroid):
    return np.dot(point, centroid) / (np.sqrt(np.sum(point ** 2)) * np.sqrt(np.sum(centroid ** 2)))


def assign_label(distance, data_point):
    index = max(distance, key=distance.get)
    return [index, data_point]


def new_centroids(point, centroid):
    return np.array(point + centroid) / 2


def k_means(data_points, centroids, max_iteration):
    for i in range(0, max_iteration):
        cluster_label = []
        temp_centroids = centroids
        for j in range(0, len(data_points)):
            distance = {}
            for k in range(0, len(centroids)):
                distance[k] = cosine_similarity(data_points[j], centroids[k])
            label = assign_label(distance, data_points[j])
            centroids[label[0]] = new_centroids(label[1], centroids[label[0]])
            cluster_label.append(label)
        if np.array_equal(np.array(temp_centroids), np.array(centroids)):
            break
    return cluster_label, centroids


def init_centroids(points):
    centroids = []
    for i in range(0, len(points)):
        centroids.append(points[i])
    return np.array(centroids)


def output_result(label_data, centroids):

    print("After Clustering:")

    for i in label_data:
        print("data point:", i[1], "cluster number: ", i[0])

    print("After clustering centroids position: ", centroids)


def load():
    d_vec = np.loadtxt('d_vec.csv', delimiter=',')
    return d_vec


def main():

    data_points = load()
    centroids = init_centroids([data_points[0], data_points[200], data_points[400], data_points[550], data_points[660]])
    max_iteration = 100
    label_data, last_centroids = k_means(data_points, centroids, max_iteration)
    output_result(label_data, last_centroids)


if __name__ == '__main__':
    main()