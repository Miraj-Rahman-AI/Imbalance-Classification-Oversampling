import numpy as np
from collections import Counter
import warnings

# Suppress warnings to avoid clutter in the output
warnings.filterwarnings("ignore")

# Minimum number of samples required to split a node
minNumSample = 10

class BinaryTree:
    """A Special Binary Tree for storing data and labels.

    This class represents a binary tree where each node stores data, labels, and references
    to its left and right children.

    Attributes:
        label (np.array): Labels associated with the node.
        data (np.array): Data stored in the node.
        leftChild (BinaryTree): Reference to the left child node.
        rightChild (BinaryTree): Reference to the right child node.
    """

    def __init__(self, labels=np.array([]), datas=np.array([])):
        self.label = labels
        self.data = datas
        self.leftChild = None
        self.rightChild = None

    def set_rightChild(self, rightObj):
        """Set the right child of the node."""
        self.rightChild = rightObj

    def set_leftChild(self, leftObj):
        """Set the left child of the node."""
        self.leftChild = leftObj

    def get_rightChild(self):
        """Get the right child of the node."""
        return self.rightChild

    def get_leftChild(self):
        """Get the left child of the node."""
        return self.leftChild

    def get_data(self):
        """Get the data stored in the node."""
        return self.data

    def get_label(self):
        """Get the label associated with the node."""
        return self.label


def RSDS(train_data, tree_num=100):
    """Handling data noise using completely random forest judgment.

    This function builds a forest of completely random trees and uses them to identify and remove
    noisy data points from the training set.

    Parameters:
        train_data (np.array): The training dataset.
        tree_num (int): The number of random trees to build.

    Returns:
        np.array: The denoised training dataset.
    """

    m, n = train_data.shape
    forest = np.array([])
    for i in range(tree_num):
        tree = CRT(train_data)
        visiTree = visitCRT(tree)  # Shape (2, n)
        visiTree = visiTree[:, np.argsort(visiTree[0, :])]
        visiTree = visiTree[1, :]  # Get labels
        if forest.size == 0:
            forest = visiTree.reshape(m, 1)
        else:
            forest = np.hstack((forest, visiTree.reshape(m, 1)))
    
    noiseForest = np.sum(forest, axis=1)
    nn = 0.5 * tree_num
    # Original algorithm only retains boundary points, removes noise points and safe points
    noiseForest = np.array(list(map(lambda x: 1 if x >= nn or x == 0 else 0, noiseForest)))
    denoiseTraindata = deleteNoiseData(train_data, noiseForest)
    return denoiseTraindata


def CRT(data):
    """Build A Completely Random Tree.

    This function builds a completely random tree by recursively splitting the data based on
    randomly selected attributes and split values.

    Parameters:
        data (np.array): The dataset to build the tree from.

    Returns:
        BinaryTree: The root of the completely random tree.
    """

    numberSample = data.shape[0]
    orderAttribute = np.arange(numberSample).reshape(numberSample, 1)  # (862, 1)
    data = np.hstack((data, orderAttribute))
    completeRandomTree = generateTree(data)
    return completeRandomTree


def generateTree(data, uplabels=[]):
    """Iteratively Generating A Completely Random Tree.

    This function recursively generates a completely random tree by randomly selecting attributes
    and split values to partition the data.

    Parameters:
        data (np.array): The dataset to split.
        uplabels (list): The labels from the parent node.

    Returns:
        BinaryTree: The root of the generated tree.
    """

    try:
        numberSample, numberAttribute = data.shape
    except ValueError:
        numberSample = 1
        numberAttribute = data.size

    if numberAttribute == 0:
        return None

    numberAttribute = numberAttribute - 2  # Subtract the added serial number and label

    # The category of the current data, also called the node category
    labelNumKey = []  # todo
    if numberSample == 1:  # Only one sample left
        labelvalue = data[0][0]
        rootdata = data[0][numberAttribute + 1]
    else:
        labelNum = Counter(data[:, 0])
        labelNumKey = list(labelNum.keys())  # Key (label)
        labelNumValue = list(labelNum.values())  # Value (quantity)
        labelvalue = labelNumKey[labelNumValue.index(max(labelNumValue))]  # Vote to find the label
        rootdata = data[:, numberAttribute + 1]
    rootlabel = np.hstack((labelvalue, uplabels))  # todo

    # Call the class 'BinaryTree', passing in tags and data
    CRTree = BinaryTree(rootlabel, rootdata)

    # There are at least two conditions for the tree to stop growing:
    # 1 the number of samples is limited;
    # 2 the first column is all equal
    if numberSample < minNumSample or len(labelNumKey) < 2:
        # minNumSample defaults to 10 or only 1 of the label types are left.
        return CRTree
    else:
        maxCycles = 1.5 * numberAttribute  # Maximum number of cycles
        i = 0
        while True:
            i += 1
            splitAttribute = np.random.randint(1, numberAttribute)  # Randomly select a list of attributes
            if splitAttribute > 0 and splitAttribute < numberAttribute + 1:
                dataSplit = data[:, splitAttribute]
                uniquedata = list(set(dataSplit))
                if len(uniquedata) > 1:
                    break
            if i > maxCycles:  # Tree caused by data anomaly stops growing
                return CRTree
        sv1 = np.random.choice(uniquedata)
        i = 0
        while True:
            i += 1
            sv2 = np.random.choice(uniquedata)
            if sv2 != sv1:
                break
            if i > maxCycles:
                return CRTree
        splitValue = np.mean([sv1, sv2])

        # Call split function
        leftdata, rightdata = splitData(data, splitAttribute, splitValue)

        # Set the left subtree, the right subtree
        CRTree.set_leftChild(generateTree(leftdata, rootlabel))
        CRTree.set_rightChild(generateTree(rightdata, rootlabel))
        return CRTree


def visitCRT(tree):
    """Traversing the tree to get the relationship between the data and the node label.

    This function traverses the tree and stores the data number and node label stored in each node
    of the completely random tree.

    Parameters:
        tree (BinaryTree): The root node of the tree.

    Returns:
        np.array: A matrix of two rows and N columns, the first row is the index of the sample,
                  and the second row is the threshold of the label noise.
    """

    if not tree.get_leftChild() and not tree.get_rightChild():  # If the left and right subtrees are empty
        data = tree.get_data()  # data is the serial number of the sample
        labels = checkLabelSequence(tree.get_label())  # Existing tag sequence
        try:
            labels = np.zeros(len(data)) + labels
        except TypeError:
            pass
        result = np.vstack((data, labels))
        return result
    else:
        resultLeft = visitCRT(tree.get_leftChild())
        resultRight = visitCRT(tree.get_rightChild())
        result = np.hstack((resultLeft, resultRight))
        return result


def deleteNoiseData(data, noiseOrder):
    """Delete noise points in the training set.

    This function deletes the noise points in the training set according to the noise
    judgment result of each data in noiseOrder.

    Parameters:
        data (np.array): The training dataset.
        noiseOrder (np.array): A list indicating whether each data point is noise.

    Returns:
        np.array: The denoised dataset.
    """

    m, n = data.shape
    data = np.hstack((data, noiseOrder.reshape(m, 1)))
    redata = np.array(list(filter(lambda x: x[n] == 0, data[:, ])))
    redata = np.delete(redata, n, axis=1)
    return redata


def checkLabelSequence(labels):
    """Check label sequence.

    This function checks if the leaf node label is the same as the parent node label.

    Parameters:
        labels (np.array): The label sequence.

    Returns:
        int: 1 if the labels are different, 0 otherwise.
    """

    return 1 if labels[0] != labels[1] else 0


def splitData(data, splitAttribute, splitValue):
    """Dividing data sets.

    This function divides the data into two parts, leftData and rightData, based on the splitValue
    of the split attribute column element.

    Parameters:
        data (np.array): The dataset to split.
        splitAttribute (int): The attribute to split on.
        splitValue (float): The value to split the attribute on.

    Returns:
        tuple: A tuple containing the left and right datasets.
    """

    rightData = np.array(list(filter(lambda x: x[splitAttribute] > splitValue, data[:, ])))
    leftData = np.array(list(filter(lambda x: x[splitAttribute] <= splitValue, data[:, ])))
    return leftData, rightData


def RSDS_smote(train_data, tree_num=100):
    """RSDS for multi-class imbalanced datasets.

    This function is an improved version of RSDS for handling multi-class imbalanced datasets.
    It first denoises the data, then compresses the majority class by removing internal points.

    Parameters:
        train_data (np.array): The training dataset.
        tree_num (int): The number of random trees to build.

    Returns:
        np.array: The processed dataset.
    """

    m, n = train_data.shape
    forest = np.array([])
    for i in range(tree_num):
        tree = CRT(train_data)
        visiTree = visitCRT(tree)  # Shape (2, n)
        visiTree = visiTree[:, np.argsort(visiTree[0, :])]
        visiTree = visiTree[1, :]  # Get labels
        if forest.size == 0:
            forest = visiTree.reshape(m, 1)
        else:
            forest = np.hstack((forest, visiTree.reshape(m, 1)))
    
    noiseForest = np.sum(forest, axis=1)
    print(noiseForest)

    # Step 1: Denoise the data, retaining boundary points and internal points
    nn = 0.5 * tree_num  # Threshold
    innerForest = np.array(list(map(lambda x: 1 if x == 0 else 0, noiseForest)))  # Internal points
    noiseForest = np.array(list(map(lambda x: 1 if x >= nn else 0, noiseForest)))  # Noise points
    print(Counter(innerForest), '\n', Counter(noiseForest))

    denoiseTraindata = deleteNoiseData(train_data, noiseForest)  # Retain points where noiseForest is 0
    innerForest_denoise = innerForest[np.where(noiseForest == 0)]  # Boundary points after denoising
    print('Denoised data count:\t', len(denoiseTraindata), innerForest_denoise.shape, '\n')

    # Step 2: Compress the majority class by recursively removing internal points
    time = 0  # Compression count
    num_dict = Counter(denoiseTraindata[:, 0])  # Dict: label -> count after denoising
    max_times = len(num_dict)  # Maximum compression count

    def step2(denoiseTraindata, times: int, num_dict: dict, max_times: int):
        """Recursively compress the majority class.

        Parameters:
            denoiseTraindata (np.array): The denoised dataset.
            times (int): The current compression count.
            num_dict (dict): A dictionary of label counts.
            max_times (int): The maximum number of compressions.

        Returns:
            np.array: The compressed dataset.
        """

        major_labels = max(num_dict, key=num_dict.get)  # The label of the majority class
        major_inner_Forest = innerForest_denoise[np.where(denoiseTraindata[:, 0] == major_labels)]  # Majority class boundary points

        major_data = denoiseTraindata[np.where(denoiseTraindata[:, 0] == major_labels)]  # Denoised majority class
        major_data_deinner = major_data[np.where(major_inner_Forest == 0)]  # Remove internal points

        train_data = np.vstack([
            major_data_deinner,
            denoiseTraindata[np.where(denoiseTraindata[:, 0] != major_labels)]
        ])  # Merge datasets
        times += 1

        new_major_labels = max(Counter(train_data[:, 0]), key=Counter(train_data[:, 0]).get)  # New majority class

        if new_major_labels == major_labels:
            return train_data
        elif new_major_labels != major_labels and times < max_times:
            del num_dict[major_labels]
            return step2(train_data, times, num_dict, max_times)  # Recursive compression
        elif times == max_times:
            return train_data

    result = step2(denoiseTraindata, time, num_dict, max_times)
    return result


def RSDS_feb(train_data, tree_num=100):
    """Handling data noise using completely random forest judgment.

    This function is similar to RSDS but returns both the denoised dataset and the weights
    associated with each data point.

    Parameters:
        train_data (np.array): The training dataset.
        tree_num (int): The number of random trees to build.

    Returns:
        tuple: A tuple containing the denoised dataset and the weights.
    """

    m, n = train_data.shape
    forest = np.array([])
    for i in range(tree_num):
        tree = CRT(train_data)
        visiTree = visitCRT(tree)  # Shape (2, n)
        visiTree = visiTree[:, np.argsort(visiTree[0, :])]
        visiTree = visiTree[1, :]  # Get labels
        if forest.size == 0:
            forest = visiTree.reshape(m, 1)
        else:
            forest = np.hstack((forest, visiTree.reshape(m, 1)))
    
    orignal_Forest = np.sum(forest, axis=1)  # Weights before denoising
    nn = 0.5 * tree_num  # Threshold

    # Only remove noise points, retain boundary and internal points. 1: true, 0: false
    noiseForest = np.array(list(map(lambda x: 1 if x >= nn else 0, orignal_Forest)))  # Noise points
    weight_denoise = orignal_Forest[np.where(noiseForest == 0)]  # Weights after denoising

    denoiseTraindata = deleteNoiseData(train_data, noiseForest)  # Retain points where noiseForest is 0
    return denoiseTraindata, weight_denoise  # Denoised data and weights