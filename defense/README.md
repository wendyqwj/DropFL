## Defenses
**1. Abnormal detection**\
We assume that the server is trusted and propose a statistical-based outlier detection method to flag clients that may be compromised by an attacker. Specifically, the server maintains a list of the health values of all clients in each communication round, which can be represented by the online rate. The ideal value of the online rate of each client is equal to the fraction of clients selected by the server C in federated learning. The key idea is to rank all clients according to their online rate value and flag the clients with the lowest values, which may be the compromised clients.

<div align = center>
<img src ="https://github.com/wendyqwj/DropFL/blob/main/defense/img/fig_detection_00.png" class ="center" width="210px"> 
</div>

[Evaluation results]: We set the fraction of compromised clients to 30%. Fig. 7(a) shows that the detection rate of compromised clients increases as the fraction of inspected clients increases. Our proposed method has the best detection performance in the full dropout attack, the second in the SV-based dropout, and the worst in the random dropout. For instance, when the fraction of inspected clients is 20%, that is the last 20% of clients in the list of online rate, the detection rate of compromised clients exceeds 90% in the full dropout attack.

**2. Mitigating attack**\
Although the server can effectively flag the compromised clients by the outlier detection method, it is not enough to mitigate the impact of our proposed attacks. Learning more private data from the client in a constrained communication environment is challenging. Therefore, we further propose a selection strategy based on the method of abnormal detection (ADS) for the server instead of a random selection strategy. Concretely, in the client selection step, the server first updates the health value for each client in the t-th communication round, by which the server ranks the clients in descending order. Then, all clients are evenly divided into group A and group B. The health values of clients in Group A are higher than those in Group B. The server selects mAt clients from group A with a biased probability λ, and mBt clients from group B with a probability of 1 − λ, where m^A_t = max(K · C · λ, 1), m^A_t = max(K · C · (1 − λ), 1) and 0 < λ < \frac{1}{2C}. In this way, the selection strategy not only ensures that honest clients participate in FL as much as possible, but also that the private data of compromised clients can be learned with a certain probability.

<div align =center>
<img src ="https://github.com/wendyqwj/DropFL/blob/main/defense/img/fig_mnist_mlp_ads_00.png" class ="center" width="210px"><img src ="https://github.com/wendyqwj/DropFL/blob/main/defense/img/fig_cifar_cnn_ads_00.png" class ="center" width="210px"><img src ="https://github.com/wendyqwj/DropFL/blob/main/defense/img/fig_cifar100_cnn_ads_00.png" class ="center" width="210px">
</div>

[Evaluation results]: Fig. 7(b)-(d) compares the performance of our dropout attack with non-attack using various biased probabilities under the ADS strategy. For instance, when we set the biased probability to 80%, the defense against our random dropout attack is effective. In contrast, the test accuracy of our SV-based dropout and full dropout slightly decreases.
