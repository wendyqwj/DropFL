Defenses
A.1 Abnormal detection
We assume that the server is trusted and propose a statistical-based outlier detection method to flag clients that may be compromised by an attacker. Specifically, the server maintains a list of the health values of all clients in each communication round, which can be represented by the online rate. The ideal value of the online rate of each client is equal to the fraction of clients selected by the server C in federated learning. The key idea is to rank all clients according to their online rate value and flag the clients with the lowest values, which may be the compromised clients.
![fig_detection_00](https://github.com/wendyqwj/DropFL/assets/105483021/8bcec20d-d306-4150-8bd7-a2f7e0dc6bf7)
