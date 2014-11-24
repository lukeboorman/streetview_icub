import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import unittest
import numpy as np
import random
from locationLuke import Location


#Use a seed so results are consistent
SEED=2942875  #95731  #73765
random.seed(SEED)    #careful, these are different RNGs!
np.random.seed(SEED)
unittesting=0

class DGHelper:
    def __init__(self, numOfSurfFeatures=None, initialECDGweights=None, initialCA3CA1weights=None, X=3, N=4, initialSemantics=None):
        self.X=X
        self.N=N
        self.numOfEncodedFeatures = self.X*self.N
        self.numOfSurfFeatures = numOfSurfFeatures
        #Train the network on the clean initial data
        if initialECDGweights is None:
            self.ECDGweights = np.random.rand(N, X, X)/10
        else:
            self.ECDGweights = initialECDGweights

        if initialCA3CA1weights is None:
            self.CA3CA1weights = np.zeros(shape=(N, X, X))
        else:
            self.CA3CA1weights = initialCA3CA1weights

        if initialSemantics is None:
            if self.numOfSurfFeatures is not None:
                self.generateSemantics(N,X,self.numOfSurfFeatures)
            else:
                raise NameError("If semantics are not supplied the we must know the number of surf features to generate semantics!")
        else:
            self.semanticIndices = initialSemantics

        #print("ECDG:\n%s\nCA3CA1:\n%s" % (self.ECDGweights, self.CA3CA1weights))


    def getOriginalValues(self, thresholdedFeatureVector):
        #Use the (sparse) feature vector (which has been dumb decoded so we know which input features should infact be active
        #and the semantics we derived originally to get the input values that should be active
        #Decode using same semantics originally chosen
        activeSURFIndices = np.array(self.semanticIndices[thresholdedFeatureVector])
        decoded = np.zeros((self.numOfSurfFeatures,), dtype=np.int8) #THIS COULD BE WRONG
        decoded[activeSURFIndices] = 1
        return decoded

    def getSemanticValues(self, featureVector):
        #Use the semantic indices decided upon initialisation of this DGState what the feature vector should look like.
        return np.array(featureVector[self.semanticIndices])

    def generateSemantics(self, N, X, numOfFeatures):
        #This one avoids duplicate SURF features being used in the same block
        self.semanticIndices = np.zeros((N,X), np.int8)
        for blockInd, block in enumerate(self.semanticIndices):
            self.semanticIndices[blockInd] = random.sample(xrange(numOfFeatures), X)

    def encode(self, inputActivationValues):
        #Dot product the ECDGweights with their activation to give activation values between -1 and 1
        outputActivationValues = np.zeros(shape=inputActivationValues.shape)

        #A block is a page of a 3d matrix
        for blocknum, block in enumerate(self.ECDGweights):
            outputActivationValues[blocknum] = np.dot(block, inputActivationValues[blocknum])

        #Output activations have the form [page1[ outputactivation0, ouputactivation1, outputactivation2],
        #                                  page2[ outputactivation0, outputactivation1, outputactivation2]] i.e columns are the output units, rows are the blocks
        #print("ouputActivation after:\n%s" % np.around(outputActivationValues, 3))
        
        #Smart collapse the whole matrix to get the winner
        encodedValues = smartCollapseMatrix(outputActivationValues)
        return encodedValues

    def decode(self, probabilitiesOfFiring):
        #Use "grey values" coming out of boltzmann to calculate the winners using smart collapse
        #These are the probabilities that the OUTPUT units of the sparse repreentation are on., since only one can be on at a time we do a smart collapse
        probsReshaped = probabilitiesOfFiring.reshape(self.N,self.X)
        #print("Probabilities reshaped:\n%s" % probsReshaped)
        winningNeurons = smartCollapseMatrix(probsReshaped)
        probabilityOfActivation = np.zeros(winningNeurons.shape)

        #Apply transpose of ECDGweights to reverse the effect (i.e. calculate which inputs should be on given that an output is on
        #for blocknum, block in enumerate(self.ECDGweights):
        for blocknum, block in enumerate(self.CA3CA1weights):
            #print("transpose ECDGweights:\n%s" % np.transpose(block))
            #print("winning output neurons:\n%s" % winningNeurons[blocknum])
            #We transpose the matrix as this allows us to see, given that output X is on, what are the probabilites that input units A,B,C are on

            #TODO: Instead of using transpose of original ECDGweights, use the ECDGweights learnt by perceptron
            #probabilityOfActivation[blocknum] = np.dot(np.transpose(block), winningNeurons[blocknum])
            probabilityOfActivation[blocknum] = np.dot(block, winningNeurons[blocknum])

        #print("Probability of activation after ECDGweights have been applied:\n%s" % probabilityOfActivation)
        #We now have the probability that each feature is present, dumb decode it, i.e. if its still more than 50% likely to be on, then count it as on
        thresholded = (probabilityOfActivation>=0.5)
        
        #Decode using same semantics originally chosen
        decoded = self.getOriginalValues(thresholded)
        return decoded

    def setECDGWeights(self, ECDGweights):
        self.ECDGweights = ECDGweights

    def setCA3CA1Weights(self, CA3CA1weights):
        self.CA3CA1weights = CA3CA1weights

    def learn(self, inputActivationValues, learnCA3CA1weights=False, learningrate=0.01):
        #Get semantic values for input
        sv = self.getSemanticValues(inputActivationValues)
        
        #Winning neurons are the DG output
        winningNeurons = self.encode(sv)

        #We only want to learn one set of weights at a time
        if learnCA3CA1weights:
            self.learnCA3CA1weights(inputActivationValues, winningNeurons, learningrate)
        else:
            self.learnECDGweights(winningNeurons, sv, learningrate)

    def learnECDGweights(self, winningNeurons, semanticValues, learningrate=0.01):
        """
        Winner takes all learning between EC and DG representations.
        inputActivationValue is the activation coming out of the EC, currently this is a boolean vector
        of whether the image has SURF features matching common ones discovered in the SURFExtraction phase.
        """

        #Give all none active neurons a negative activation to introduce negative ECDGweights
        #winningNeurons = (winningNeurons==0).choose(winningNeurons,-0.01)

        N = winningNeurons.shape[0]
        X = winningNeurons.shape[1]

        #This uses broadcasting to create a tiled transpose of winningNeurons,
        #Tiling converts a winning neuron activation (say neuron 2 won) [0, 0, 1] to the changes to be made to every weight,
        #I.e because neuron 2 one, all the connections to this neuron should be increased for this block, i.e. 
        #[[0, 0, 0]
        # [0, 0, 0]
        # [1, 1, 1]] since rows are output units and columns input units in the weight representation

        #Otherwise have the winning output increase connections from all inputs to it
        #New axis required as otherwise broadcasting wont work, i.e. because its trying to broadcast (4,3) onto (4,3,1)
        self.ECDGweights += (learningrate*(winningNeurons.reshape(N,X,1)))*semanticValues[:, np.newaxis]

        #Normalise weights row by row (add up all elements of each row and divide each value by that number
        self.ECDGweights = normalise(self.ECDGweights,2)

    def learnCA3CA1weights(self, inputActivationValues, DGEncodedValues, learningrate=0.01):
        #Alter the encoded values given by the ECDGweights learnt going from EC-DG to account for the new data

        #We now know both the optimum output of the Boltzmann machine after winner take all has been done (after smart collapse) - only if the data wasn't noisy in the first place? - 
        # the collapsed output (sparse)
        # The optimum output of the boltzmann machine once smart collapse has been applied would be the original input to it, if the data is clean.
        # 0 0 0 1 0 0 1 0 1 0 0 0
        # fully connected to the output which knows whether it should be on or off (the original data if trained with clean data). If one is on and the other is on, increase the ECDGweights between them?

        #Ideally we would do offline learning? Be given a list of all the inputActivationValues, and all the correctOutputActivationValues and get our error down below a threshold? 
        """
        print("CA3CA1weights:\n%s" % self.CA3CA1weights)
        print("inputActivationValues:\n%s" % inputActivationValues)
        print("DGEncodedValues:\n%s" % DGEncodedValues)
        """
        
        #Threshold = bias?
        threshold = 0.5
        givenOutputPerBlock = np.zeros(shape=DGEncodedValues.shape)
        #thresholdedOutput at clipped at 0.5 is equivalent to bias?
        thresholdedOutput = np.zeros(shape=DGEncodedValues.shape, dtype=bool)

        #print("CA3CA1weights:\n%s" % np.around(self.CA3CA1weights, 3))
        for blocknum, block in enumerate(self.CA3CA1weights):
            #print("encoding CA1CA3 block:\n%s" % block)
            #print("input:\n%s" % inputActivationValues[blocknum])
            #print("OutputActivation:\n%s" % np.dot(block, DGEncodedValues[blocknum]))
            givenOutputPerBlock[blocknum] = np.dot(block, DGEncodedValues[blocknum])
            #givenOutputPerBlock needs to be changed back into a input vector by use of its semantics its encoded in
        thresholdedOutput = (givenOutputPerBlock>=threshold)

        #Bit of a hack, go from the calculated output of the CA3 representation to the CA1 representation:
        #Get the original values (giving the representation in CA1 form) by relating the CA3 to the semantics initially decided
        CA1Form = self.getOriginalValues(thresholdedOutput)
        #print("CA1Form:\n%s" % CA1Form)

        #Get the DG representation of this output (the decoded from boltzmann machine) so we can use the perceptron learning rule on it (compare the desired output with the real output)
        realOutput = self.getSemanticValues(CA1Form)
        #Get the desired output by getting this form from the input activation (i.e. We know it if the decode was perfect it should be the same as the original input
        desiredOutput = self.getSemanticValues(inputActivationValues)

        #Use Perceptron learning algorithm to change the weights in the direction of errors
        #NOTE: Should something be transposed here as the weights are from sparse to SURF-Features not the other way round?
        difference = desiredOutput - realOutput

        N = difference.shape[0]
        X = difference.shape[1]
    
        #FIXME: Definitely not sure if DGEncodedValues is the right thing... just a guess
        changesInWeights = ((learningrate*(difference.reshape(N,X,1))*DGEncodedValues[:, np.newaxis]))

        self.CA3CA1weights += changesInWeights

        #Normalise
        #self.CA3CA1weights = normalise(self.CA3CA1weights,2)

class DGState:
#    def __init__(self,  ec, dictGrids, dghelper=None, N_place_cells=13):
    def __init__(self,  ec, dictGrids, dghelper=None): # Luke modified....
        self.N_place_cells=len(dictGrids.dictPlace) # N_place_cells
        self.dghelper = dghelper
        #HOOK:, needs to use EC data to define "combis" of features aswell

        if dghelper is not None:
            #Lets say for now that place whisker combos etc are all encoded normally, and SURF features are encoded using WTA DG. In the end we may make sure that we have blocks referring only to location, blocks refering only to whiskers, blocks refering only to light, etc.
            #FIXME: This needs changing when integrated to just get the number of surf features from ec!
            if unittesting:
                #Slice the SURF features from the numpy array
                self.numOfSurfFeatures = len(ec)
                self.surfFeatures = ec[-self.numOfSurfFeatures:]

            else:
                self.numOfSurfFeatures = len(ec.surfs)
                self.surfFeatures = ec.surfs 
            

            #Choose semantics by choosing X random features N times to make N blocks
            #For now be stupid, allow the same combinations to come up and the same indices to be compared with each other for winner take all (will the conflict break it?)
            #Make this more intelligent later
            #Make random windows associated with the features, i.e. for N windows, choose X random features to encode, make a matrix with the blocks and values
            #       <---X--->
            #    +-------------+
            # ^  | 0 0 0 0 1 0 |
            # |  | 1 0 0 0 0 0 |
            # N  |             |
            # |  |             |
            # |  |             |
            #    +-------------+


            self.semanticValues = dghelper.getSemanticValues(self.surfFeatures)

            #These are our input activations, once passed through a neural network with competitive learning applied to its ECDGweights to encourage winner takes all, the output should only have 1 active value per block (row), thus is sparse
            #What happens if none of the features are active?? Should the one with the highest weight win? Or should there just be no activation in that block making it a even sparser matrix? I suspect the latter!
            self.encode()

        if not unittesting:
            if dghelper is None:
                self.encodedValues = np.array([])
            #TODO: Need to remove place cells.... 
            #self.N_place_cells = 13
            # N_hd = 4       
            loc=Location(dictGrids) #NEW, pure place cells in DG # Luke added in Nmax
            loc.setGrids(ec.grids)
            #print str(ec.grids)
            self.place=np.zeros(self.N_place_cells)         
            self.place[loc.placeId] = 1

            self.hd_lightAhead = np.zeros(4)
            if ec.lightAhead == 1:
                self.hd_lightAhead = ec.hd.copy()

            self.whisker_combis = np.zeros(3)  #extract multi-whisker features. 
            self.whisker_combis[0] = ec.whiskers[0] * ec.whiskers[1] * ec.whiskers[2]   #all on
            self.whisker_combis[1] = (1-ec.whiskers[0]) * (1-ec.whiskers[1]) * (1-ec.whiskers[2])   #none on
            self.whisker_combis[2] = ec.whiskers[0] * (1-ec.whiskers[1]) * ec.whiskers[2]   # both LR walls but no front

    def toVectorSurfOnly(self):
        if len(self.encodedValues) == 0:
            return self.encodedValues 
        else:
            return np.hstack((self.encodedValues.flatten()))

    def toVector(self):
        return np.hstack((self.place.flatten(), self.hd_lightAhead, self.whisker_combis, self.encodedValues.flatten()))

    def toVectorSensesOnly(self):
        return np.hstack((self.whisker_combis, self.toVectorSurfOnly()))
        #return np.hstack((self.whisker_combis, self.encodedValues.flatten()))

    def toVectorOdomOnly(self):
        return np.hstack((self.place.flatten(), self.hd_lightAhead))

    def smartCollapse(self):
        self.place = smartCollapse(self.place)

    def encode(self):
        self.encodedValues = self.dghelper.encode(self.semanticValues)
        
    def decode(self, probabilitiesOfFiring):
        self.decodedValues = self.dghelper.decode(probabilitiesOfFiring)
        return self.decodedValues

def smartCollapseMatrix(xs):
    #Use of argmax gives a maximum value no matter what, if a block is [0,0,0,0] the first index will be chosen as the maximum, this may not be desirable
    idx = np.argmax(xs, 1)
    r = np.zeros(xs.shape, np.int8)
    for row, col in enumerate(idx):
        r[row, col] = 1
    return r 

def smartCollapse(xs):
    idx=np.argmax(xs)
    r = np.zeros(xs.flatten().shape)
    r[idx]=1
    return r.reshape(xs.shape)

def addNoise(data, probability):
    noisyData = data.copy()
    for ind in range(len(data)):
        if random.random() < probability:
            noisyData[ind] = 1 - noisyData[ind]
    return noisyData

def accuracy(activation1, activation2):
    same = np.int8(np.logical_not(np.bitwise_xor(activation1, activation2)))
    return np.sum(same)/float(len(same))

def normalise(matrix, axis):
    rowsSummed = np.sum(matrix, axis)
    X = matrix.shape[1]
    N = matrix.shape[0]
    #If its a row normalisation (sum rows and divide by rows)
    if axis == 2:
        #print("rowsSummed:\n%s\nN:%d X:%d"% (rowsSummed, N,X))
        reshaped = np.reshape(rowsSummed, (N,X,1))
    elif axis == 1:
        reshaped = np.reshape(rowsSummed, (N,1,X))
    else:
        raise NameError("Axis must be rows or columns, axis == 2 is to add up a whole row and divide the row by that,\
            axis == 1 is to add up a whole column and divide the column by that")

    normalised = matrix / reshaped
    return normalised

def train_weights(trainingData, X, N, presentationOfData, learningrate=0.01):

    #Train the network on the clean initial data
    #initialECDGWeights = np.random.rand(N, X, X)/10
    #initialCA3CA1Weights = np.zeros(shape=(N, X, X))
    numOfSurfFeatures = len(trainingData[0])
    
    dgh = DGHelper(numOfSurfFeatures,X=X,N=N)
    
    #def __init__(self,  ec, dictGrids, dghelper):
    #trainingdg = DGState(trainingData[0], None, dgh)
    for x in range(presentationOfData):
        for data in trainingData:
            dgh.learn(data, False, learningrate)
    for x in range(presentationOfData):
        for data in trainingData:
            dgh.learn(data, True, learningrate)
    #Since the data has no patterns this might not work...

    return dgh

def calculate_performance(trainingData, inputDataSet, X, N, presentationOfData, learningrate=0.01):
    numOfImages = inputDataSet.shape[0]

    dgh = train_weights(trainingData, X, N, presentationOfData, learningrate)

    #Feed noisy data through EC-DG
    encodedData = np.zeros((numOfImages,N,X), dtype=np.int8)
    for imageNum, data in enumerate(inputDataSet):
        testingdg = DGState(data, None, dgh)
        encodedData[imageNum] = testingdg.encodedValues

    #pass DG onto CA1 as if it was the collapsed data, 
    decodedData = np.zeros(inputDataSet.shape, dtype=np.int8)
    for imageNum, data in enumerate(encodedData):
        decodedData[imageNum] = dgh.decode(data)

    #Compare CA1 and non-noisy EC
    #Performance is an XNOR between the two codes before and after noise
    totalAccuracy = 0
    for imageNum, origData in enumerate(trainingData):
        totalAccuracy += accuracy(origData, decodedData[imageNum])
    totalAccuracy = totalAccuracy/float(inputDataSet.shape[0])*100

    #Calculate how much change the encode and decode has made (difference between noisy EC and CA1
    totalChange = 0
    for imageNum, noisyData in enumerate(inputDataSet):
        totalChange += accuracy(noisyData, decodedData[imageNum])
    totalChange = (1-(totalChange/float(inputDataSet.shape[0])))*100
    return totalAccuracy, totalChange

class TestEncoding(unittest.TestCase):
    def setUp(self):
        #Use a seed so results are consistent
        SEED=2942875  #95731  #73765
        random.seed(SEED)    #careful, these are different RNGs!
        np.random.seed(SEED)

        #Make fake data and noisy copy
        self.fakeSURF = np.random.randint(0,2, (10,))
        noiseProb = 0.1

        self.noisyFakeSURF = addNoise(self.fakeSURF, noiseProb)

        self.N=4
        self.X=3
        #Choose the noisy data (so we're not relying on the data seed to make this work!)
        #self.chosenNoisyData = np.array([1, 0, 0, 0, 0, 0, 1, 1, 0, 1])
        self.chosenNoisyData = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        self.semantics = np.array([[2,1,7],[0,5,2],[5,2,8],[3,5,0]])

        #Test ECDGweights
        #Weights are as so where sf = surf feature directly from the input of ec, and ou = output unit which is the neuron that will fire in response to winner takes all
        #      /                   /
        #     /___________________/ b2
        #    /                   /
        #   /___________________/ b1
        #   |sf0____sf1_____sf2_|
        #ou0| x      x       x  |
        #   |                   |
        #ou1| x      x       x  |
        #   |                   | |/ 
        #ou2| x      x       x  | |
        #   |___________________|/

        self.encodedTestWeights = np.zeros(shape=(self.N,self.X,self.X))
        self.encodedTestWeights[2,0,1] = 0.5
        self.encodedTestWeights[2,0,2] = 0.3
        self.encodedTestWeights[2,2,2] = 0.75
        self.encodedTestWeights[0,0,0] = 0.5
        self.encodedTestWeights[0,1,0] = 0.75
        self.encodedTestWeights[0,2,2] = 0.25

        self.CA3CA1TestWeights = np.zeros(shape=(self.N, self.X, self.X))

        #Make dentate gyrus
        self.dgh = DGHelper(numOfSurfFeatures=len(self.chosenNoisyData), initialECDGweights=self.encodedTestWeights.copy(), initialCA3CA1weights=self.CA3CA1TestWeights.copy(),  X=self.X, N=self.N, initialSemantics=self.semantics)
        self.dg = DGState(self.chosenNoisyData, None, self.dgh)

    def test_semantic_values(self):
        data = np.array([1,0,1,0,1,0,1,0,1,0])
        semantics = np.array([[2,1,7],[0,5,2],[5,2,8],[3,5,0]])
        svdgh = DGHelper(initialSemantics = semantics)
        sv = svdgh.getSemanticValues(data)
        realSemanticValues = np.array([[1,0,0],[1, 0, 1],[0,1,1],[0,0,1]]) 
        self.assertTrue(np.all(sv == realSemanticValues))
        
    def test_making_DGState(self):
        self.assertIsNotNone(self.dg)
        self.assertIsNotNone(self.dgh)

    def test_smartCollapseMatrix(self):
        data = np.array([[1,1,2,1],[4,6,7,1],[5,2,1,1]])
        resultdata = smartCollapseMatrix(data)
        correctResultdata = np.array([[0,0,1,0],[0,0,1,0],[1,0,0,0]])
        self.assertTrue(np.all(resultdata == correctResultdata), "Winner takes all works, two matrices are equivalent")

    #@unittest.skip("Saving time whilst testing other")
    def test_dot_product(self):
        W = np.array([[200,500,0],[0,0,100],[600,0,0]])
        A = np.array([0, 1, 1])
        dotted = np.dot(W,A)
        collapsed = smartCollapse(dotted)
        """
        print("W:\n%s" % W)
        print("A:\n%s" % A)
        print("W dot A:\n%s" % dotted)
        print("W multiplied A:\n%s" % (W*A))
        print("W multiplied A and collapsed:\n%s" % smartCollapse(W*A))
        print("Smart collapse:\n%s" % collapsed) 
        print("Multiplied:\n%s" % (W*collapsed))
        print("Tile test:\n%s" % np.transpose(np.tile((np.array([0,1,0])),(3,1))))
        """
        self.assertTrue(np.all((W*A) == np.array([[0, 500, 0],[0, 0, 100],[0, 0, 0]])))

    def test_encode_type(self):
        encodedData = self.dg.toVectorSurfOnly()
        self.assertTrue(encodedData.ndim == 1)
        self.assertTrue(len(encodedData) == self.X*self.N)
        self.assertTrue(encodedData.dtype == np.int8)
        self.assertTrue(np.sum(encodedData) == self.N, "Encoded data should be sparse and only have one winner per block")

    def test_encode_ability(self):
        #Note we are still using the np.random.seed() for this to work as this is where out input navigation is coming from!
        #print("noisy data to be encoded:\n%s" % self.chosenNoisyData)
        #print("Semantics Indices:\n%s" % self.dg.semanticIndices)
        #print("Semantic values:\n%s" % self.dg.semanticValues)
        encodedData = self.dg.toVectorSurfOnly()
        #print("Encoded data:\n%s" % encodedData)
        correctResultData = np.array([0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0])
        self.assertTrue(np.all(encodedData == correctResultData), "In this case\n \
                the weight between surf feature 2 and output unit 0 block 0  has a positive weight of 0.5, activation should be 0.5 as feature 2 is active\n \
            and the weight between surf feature 2 and output unit 1 block 0  has a positive weight of 0.75, activation should be 0.75 which wins over first one\n \
            and the weight between surf feature 2 and output unit 0 block 2, has a positive weight of 0.5, activation should be 0.5 as feature 2 is activen \
            and the weight between surf feature 8 and output unit 0 block 2, has a positive weight of 0.3, activation should be 0.3 as feature 8 is active\n \
            and the weight between surf feature 8 and output unit 2 block 2, has a positive weight of 0.75, activation should be 0.75 as feature 8, however because the previous two ECDGweights both go into output unit 0, its overall activation is 0.8 which beats 0.75 thus output unit 0 wins")

    def test_learning(self):
        encodingdgh = DGHelper(numOfSurfFeatures=len(self.chosenNoisyData), initialECDGweights=self.encodedTestWeights.copy(), initialCA3CA1weights=self.CA3CA1TestWeights.copy(), X=3, N=4)
        encodingdgh.learn(self.chosenNoisyData)
        self.assertGreater(encodingdgh.ECDGweights[2,0,1], self.encodedTestWeights[2,0,1])
        self.assertGreater(encodingdgh.ECDGweights[2,0,2], self.encodedTestWeights[2,0,2])
        self.assertEqual(encodingdgh.ECDGweights[2,2,2], self.encodedTestWeights[2,2,2], "A bit fucked up because we are now normalising...")
        self.assertEqual(encodingdgh.ECDGweights[3,0,1], self.encodedTestWeights[3,0,1])
        #Since all ACTIVE input units ECDGweights connecting to the winning output are increased, this is also increased as it contributed to the units activation
        self.assertGreater(encodingdgh.ECDGweights[0,1,0], self.encodedTestWeights[0,1,0])
        self.assertEqual(encodingdgh.ECDGweights[0,0,0], self.encodedTestWeights[0,0,0])

    @unittest.skip("Saving time whilst testing other")
    def test_multiple_learning(self):
        trials = 300
        average = 20 #20
        X = 3 #4
        N = 15 #25

        accuracyOfModel = 0
        for x in range(average):
            #Chosen as a guess would give an accuracy of 50%
            initialData = np.array([0,0,0,0,0,1,1,1,1,1])
            np.random.shuffle(initialData)
            #print("initial data:\n%s" % initialData)
            #initialECDGWeights = np.random.rand(N, X, X)/10
            #initialECDGWeights = np.zeros(shape=(N, X, X))
            #CA3CA1TestWeights = np.zeros(shape=(N, X, X))
        
            #def __init_(self, initialECDGWeights=False, initialCA3CA1weights=False, X=3, N=4, initialSemantics=False):
            encodingdgh = DGHelper(numOfSurfFeatures=len(initialData), X=X,N=N)
            
            #print("Initial ECDGweights:\n%s" % (np.around(initialECDGWeights, 3)))
            
            #print("Semantics:\n%s" % encodingdg.semanticIndices)
            noiseProb = 0.1
            
            #Log activations used to look at later
            activationsUsed = np.zeros((trials,initialData.shape[0]), np.int8)
            
            #Generate data to learn with
            for trial in range(trials):
                newData = addNoise(initialData, noiseProb) 
                activationsUsed[trial] = newData
            
            #Preferable to learn in two separate phases as otherwise CA3CA1 will learn noisy mappings and slowly get better as
            #ECDG connections get better
            #Train ECDGweights
            for trial in range(trials):
                encodingdgh.learn(newData, False)
            #print("Final ECDGweights:\n%s" % (np.around(encodingdg.ECDGweights, 3)))

            #Train CA3CA1weights
            for trial in range(trials):
                encodingdgh.learn(newData, True)
                #ECDGweights = smartCollapseMatrix(encodingdg.ECDGweights)
                #print("Learnt ECDGweights after %d trials:\n%s" % (trials, encodingdg.ECDGweights))

            probabilitiesOfFiring = np.ones((1,X*N))*0.5
            encodingdg = DGState(initialData, None, encodingdgh)
            decoded = encodingdg.decode(probabilitiesOfFiring)
            #print("After ECDGweights:\n%s" % (np.around(encodingdg.ECDGweights, 3)))
            #print("Decoded:\n%s" % decoded)
            #print("Orignial:\n%s" % initialData)
            #print("All activations used:\n%s" % activationsUsed)
            accuracyOfModel += accuracy(decoded, initialData)

        #print("ECDG Weights:\n%s" % (np.around(encodingdg.ECDGweights, 3)))
        #print("CA3CA1 Weights:\n%s" % (np.around(encodingdg.CA3CA1weights, 3)))
        #print("Semantics:\n%s" % encodingdg.semanticIndices)
        #print("initialData:\n%s" % initialData)
        accuracyOfModel /= average
        accuracyOfModel = accuracyOfModel*100
        #print("Accuracy: %f%%" % accuracyOfModel)
        self.assertGreater(accuracyOfModel, 0.5, "Any less than 50% accuracy means it is worse than just guessing")
        
        #Test my equivalence of ECDG and CA3CA1 weights theory
        #self.assertTrue(np.allclose(np.around(encodingdg.ECDGweights, 3), np.around(encodingdg.CA3CA1weights, 3)))

    def test_decode(self):
        #This is how they will be originally encoded, since the probabilities of firing from the boltzmann are provided this is ignored (except for the semantics)
        #                                 2       1       7        0       5       2       5        2        8       3       5       0
        #partiallyEncoded = np.array([0.5*1, 1*0.75,  0*0.25,    1*0.25,     0*0,    1*0,    0*0,     1*0,     1*0,   0*0.5,   0*0,  1*0.75])
        #fullyEncoded = np.array([0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1])
        
        #Here are a set of possible probabilities of firings (given by a boltzmann machine)
        #These are the outputs that are winning! thats why we transpose the matrix!
        probabilitiesOfFiring = np.array([0.9, 0.2, 0.1, 0.6, 0.5, 0.1, 0.6, 0.5, 0.5, 0.05, 0.1, 0.25])
        #I.e. given that output unit 0 of block 2 is on, what are the probabilities that feature 2 and 8 are active, if they are more than 50%, they are on
        #Here are the active neurons chosen after WTA is applied
        collapsed = np.array([[1, 0, 0],
                              [1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]])
#        dg.CA1CA3weights = np.array([

        #When the output neurons activities are combined with the transpose of the ECDGweights (bringing it back to probability that each is firing?)
        outputprobabilityofunitsfiring = np.array([[ 0.5,   0.,    0.  ],
                                                   [ 0.,    0.,    0.  ],
                                                   [ 0.,    0.5,   0.3 ],
                                                   [ 0.,    0.,    0.  ]])
        #Here is how these values were collected, and their corresponding surf feature indices used for decoding (the semantics of the encoding)
        #                                             2     1    7    0    5    2    5     2      8     3    5    0 
        whereoutputsprobabilitycamefrom = np.array([1*0.5, 0*0, 0*0, 1*0, 0*0, 1*0, 0*0, 1*0.5, 1*0.3, 0*0, 0*0, 0*0])

        #Does the decoder use and AND? if its the winner of any of the competitions, then it should be on.
        #Here are the decoded values of the cleaned neurons using the probability that they will be on and their semantics to decode
        fullyDecoded = ([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])

        #Decoding should first calculate which neurons won, then convert back to EC space
        decoded = self.dgh.decode(probabilitiesOfFiring)
        #print("Should be:\n%s\nIs:\n%s" % (fullyDecoded,decoded))
        #Fails as we are no longer using ECDG transpose to decode
        self.assertTrue(np.all(decoded == fullyDecoded))

    def test_CA3CA1learning(self):
        #self.chosenNoisyData = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        #self.semantics = np.array([[2,1,7],[0,5,2],[5,2,8],[3,5,0]])

        #Weights to make feature 2 active, so the input and the output are identical, thus no weight changes
        w2on = np.array(
            [[[ 0.,    0.,    0.  ],
              [ 0.,    1.,    0.  ],
              [ 0.,    0.,    0.  ]]
              ,
             [[ 0.,    0.,    0.  ],
              [ 0.,    1.,    0.  ],
              [ 0.,    0.,    0.  ]]
              ,
             [[ 0.,    0.,    0.  ],
              [ 0.,    1.,    0.  ],
              [ 0.,    0.,    0.  ]]
              ,
             [[ 0.,    0.,    0.  ],
              [ 0.,    0.,    0.  ],
              [ 0.,    0.,    0.  ]]])

        #For learning CA3CA1 we don't need the weights for ECDG
        ecw = np.zeros(shape=(w2on.shape))

        surfFeaturesValues = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        encodingdgh = DGHelper(numOfSurfFeatures=len(surfFeaturesValues), initialECDGweights=ecw, initialCA3CA1weights=w2on.copy(), X=3, N=4, initialSemantics=self.semantics)
        #encodingdg = DGState(self.chosenNoisyData, None, encodingdgh)
        dgvalues = np.array([[1, 0, 0],
                             [0, 0, 1],
                             [0, 1, 0],
                             [0, 0, 0]])

        #sv = encodingdg.getSemanticValues(surfFeaturesValues, self.semantics)
        #The raw input should be given to CA3CA1, and it will get the semantic values OR it needs changing in that method and the sv can be given to it
        encodingdgh.learnCA3CA1weights(surfFeaturesValues, dgvalues)
        #print("PERCEPTRON ENCODED WEIGHTS:\n%s" % encodingdg.CA3CA1weights)
        self.assertTrue(np.all(w2on == encodingdgh.CA3CA1weights), "If one block's weights applied with the activation of the DG representation thinks a surffeature is present \
            it is regarded as present")

        w2off = np.array(
            [[[ 0.,    0.,    0.  ],
              [ 0.,    1.,    0.  ],
              [ 0.,    0.,    0.  ]]
              ,
             [[ 0.,    0.,    0.  ],
              [ 0.,    1.,    0.  ],
              [ 0.,    0.,    0.  ]]
              ,
             [[ 0.,    0.,    0.  ],
              [ 0.,    0.,    1.  ],
              [ 0.,    0.,    0.  ]]
              ,
             [[ 0.,    0.,    0.  ],
              [ 0.,    0.,    0.  ],
              [ 0.,    0.,    0.  ]]])

        encodingdgh = DGHelper(numOfSurfFeatures=len(surfFeaturesValues), initialECDGweights=ecw, initialCA3CA1weights=w2off.copy(), X=3, N=4, initialSemantics=self.semantics)
        #encodingdg = DGState(self.chosenNoisyData, None, ecw, w2off.copy(), X=3, N=4, semantics=self.semantics)
        encodingdgh.learnCA3CA1weights(surfFeaturesValues, dgvalues)
        #print("PERCEPTRON ENCODED WEIGHTS:\n%s" % encodingdg.CA3CA1weights)
        self.assertFalse(np.all(w2off == encodingdgh.CA3CA1weights), "If none of the block's weights applied with the activation of the DG representation think a surffeature is present \
            it is not regarded as present")

        #So if one block thinks that a neuron is on activation is over 0.5) it is counted as being active, but probabilities in combination will not work
        #I.e if block 1 thinks neuron is on with 0.25 certainty, and block 2 thinks neuron is on with 0.25 certainty, it is not concidered active
        w2halfOn = np.array(
            [[[ 0.25,  0.,    0.  ],
              [ 0.,    0.,    0.  ],
              [ 0.,    0.,    0.  ]]
              ,
             [[ 0.,    0.,    0.  ],
              [ 0.,    1.,    0.  ],
              [ 0.,    0.,    0.  ]]
              ,
             [[ 0.,    0.,    0.  ],
              [ 0.,    0.5,   1.  ],
              [ 0.,    0.,    0.  ]]
              ,
             [[ 0.,    0.,    0.  ],
              [ 0.,    0.,    0.  ],
              [ 0.,    0.,    0.  ]]])

        #encodingdg = DGState(self.chosenNoisyData, None, ecw, w2halfOn.copy(), X=3, N=4, semantics=self.semantics)
        encodingdgh = DGHelper(numOfSurfFeatures=len(surfFeaturesValues), initialECDGweights=ecw, initialCA3CA1weights=w2halfOn.copy(), X=3, N=4, initialSemantics=self.semantics)

        encodingdgh.learnCA3CA1weights(surfFeaturesValues, dgvalues)
        #print("PERCEPTRON ENCODED WEIGHTS:\n%s" % encodingdg.CA3CA1weights)
        self.assertTrue(np.all(w2halfOn == encodingdgh.CA3CA1weights), "If any of the blocks think that a surf feature is present, it is regarded as present, but not if two probabilities \
            combine, i.e. 0.25 probability of 2 being on, and 0.25 probability of it being on from two blocks")

    def test_learningCA3CA1weights(self):
        #Learn one piece of data, then encode it, and decode it
        initialData = np.array([0,0,0,0,0,1,1,1,1,1])
        X=4
        N=25
        dgh = DGHelper(numOfSurfFeatures=len(initialData), X=X, N=N)
    
        #learningDG = DGState(initialData, None, dgh)
        #print("Initial ECDGweights:\n%s" % (np.around(initialECDGWeights, 3)))
            
        #Number of learning cycles to learn the weights
        trials = 100

        #Preferable to learn in two separate phases as otherwise CA3CA1 will learn noisy mappings and slowly get better as ECDG connections get better
        #Train ECDGweights
        for trial in range(trials):
            dgh.learn(initialData, False)

        #Train CA3CA1weights
        for trial in range(trials):
            dgh.learn(initialData, True)
        
        probabilitiesOfFiring = np.ones((1,X*N))*0.5
        #encoding dg will encode the data, then this data will be decoded with the probabilities of each dg firing given
        decoded = dgh.decode(probabilitiesOfFiring)
        """
        print("Decoded:\n%s" % decoded)
        print("Orignial:\n%s" % initialData)
        print("Final ECDGweights:\n%s" % (np.around(encodingdg.ECDGweights, 3)))
        print("Final CA3CA1weights:\n%s" % (np.around(encodingdg.CA3CA1weights, 3)))
        """
        #After 100 trials the encode -- decode should be learnt
        self.assertTrue(np.all(decoded == initialData))

    def test_normalisation(self):
        w = np.array([[[0,      1,       1],
                       [1,      0,       0.25],
                       [0,      1,       0]]
                      ,
                      [[0,      1,       1],
                       [0.333,  0.333,   0.333],
                       [0.25,      0.25,    0]]])

        correctRowNormalised = np.array(
                            [[[ 0.,   0.5,    0.5], 
                              [ 0.8,    0.,   0.2],
                              [ 0.,     1.,   0. ]]
                              ,
                             [[ 0.,   0.5,    0.5],
                              [ 0.33333333,  0.33333333,  0.33333333],
                              [ 0.5,  0.5,    0.]]])

        rowNormalised = normalise(w, axis=2)

        self.assertTrue(np.allclose(rowNormalised, correctRowNormalised), "Testing normalising rows")

        correctColNormalised = np.array(
                            [[[ 0.,  0.5,  0.8,],
                              [ 1.,   0.,  0.2,],
                              [ 0.,  0.5,  0.,]]
                              ,
                             [[ 0.,  0.63171194,  0.75018755],
                              [ 0.57118353,  0.21036008, 0.24981245],
                              [ 0.42881647, 0.15792798,  0.]]])
        colNormalised = normalise(w, axis=1)
        self.assertTrue(np.allclose(colNormalised, correctColNormalised), "Testing normalising cols")

        #Must be given either rows or columns, not pages
        self.assertRaises(NameError, normalise, w, 0)

    def test_performanceMeasure(self):
        #Generate initial data
        initialData = np.array([0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1])
        noiseLevels = [0.1, 0.2, 0.5]
        figs = [plt.figure(), plt.figure()]
        for noiselevel in noiseLevels:
            numOfImages = 40
            probabilityOfNoise = noiselevel
            presentationOfData=40
            learningrate = 0.01
            dataBeforeNoise = np.zeros((numOfImages, initialData.shape[0]), dtype=np.int8)
            for image in range(numOfImages):
                np.random.shuffle(initialData)
                dataBeforeNoise[image] = initialData

            #Add noise to the data and save it as new data
            dataAfterNoise = np.zeros(dataBeforeNoise.shape, dtype=np.int8)
            for imageNum, image in enumerate(dataBeforeNoise):
                #np.random.shuffle(initialData)
                #dataAfterNoise[imageNum] = initialData
                dataAfterNoise[imageNum] = addNoise(image,probabilityOfNoise)

            print dataBeforeNoise
            print dataAfterNoise
            
            #Plot accuracy as graph
            Xs=[1,2,3,4,5,6,7,8,9,10]
            Ns=[1,5,10,15,20,25,30,35,40,45,50,60]
            A = np.zeros((len(Ns),len(Xs)))
            A = A + Xs
            B = np.zeros((len(Xs),len(Ns)))
            B = np.transpose((B + Ns))
            TA = np.zeros((A.shape[0], B.shape[1]))
            TC = np.zeros((A.shape[0], B.shape[1]))
            for xind, X in enumerate(Xs):
                for nind, N in enumerate(Ns):
                    #FIX: Bug somewhere? why doesnt accuracy decrease when data is effectively completely random, i.e. theres no correlation between initial and noisy?
                    (totalAccuracy, totalChange)= calculate_performance(dataBeforeNoise, dataAfterNoise, X, N, presentationOfData, learningrate)
                    TA[nind,xind] = totalAccuracy
                    TC[nind,xind] = 100-totalChange
                    print("Total accuracy with X=%d, N=%d, and the data being learnt over %d presentations: %f" % (X,N,presentationOfData,totalAccuracy))
                    print("Total change between noisy and decoded with X=%d, N=%d, and the data being learnt over %d presentations: %f" % (X,N,presentationOfData,totalChange))

            datatypes = [TA,TC]
            for fignum, fig in enumerate(figs):
                dataType = datatypes[fignum]
                #fig = plt.figure()
                ax = fig.gca(projection='3d')
                ax.set_color_cycle(['red', 'green', 'blue', 'yellow'])
                ax.plot_surface(A, B, dataType, rstride=1, cstride=1, alpha=0.8, cmap=plt.cm.jet,#color=(noiselevel*100,noiselevel*100,noiselevel*100), #
                                linewidth=1, antialiased=True)

                cset = ax.contour(A, B, dataType, zdir='z', offset= 0)
                cset = ax.contour(A, B, dataType, zdir='x', offset= 12)
                cset = ax.contour(A, B, dataType, zdir='y', offset= 90)

                ax.set_xlabel('X')
                ax.set_xlim3d(0, 12)
                ax.set_ylabel('N')
                ax.set_ylim3d(0, 90)
                ax.set_zlabel('Accuracy %')
                ax.set_zlim3d(0, 100)

                #Think of a nice way to plot this 3 graph for several noise levels
                if np.all(dataType == TA):
                    title = "Accuracy in de-noising noisy input when trained on clean input\nData is presented %d times with a learning rate of %f" % (presentationOfData, learningrate)
                elif np.all(dataType == TC):
                    title = "Accuracy in reconstructing noisy input when trained on clean input\nData is presented %d times with a learning rate of %f" % (presentationOfData, learningrate)
                fig.suptitle(title, fontsize=12)
        plt.show()

if __name__ == '__main__':
    unittesting = 1
    unittest.main()

