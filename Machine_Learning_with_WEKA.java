package mrzResearchArena;

/**
 * Implementation by Rafsanjani Muhammod
 * Email : rafsanjani.muhammod@gmail.com
 */


// Core import

import weka.associations.Apriori;
import weka.attributeSelection.*;
import weka.classifiers.*;
import weka.classifiers.bayes.*;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.*;
import weka.classifiers.meta.*;
import weka.classifiers.rules.*;
import weka.classifiers.trees.*;
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.Clusterer;
import weka.clusterers.Cobweb;
import weka.clusterers.SimpleKMeans;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffLoader;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.AddCluster;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.instance.RemoveWithValues;


import java.io.*;
import java.lang.*;
import java.util.*;


public class MainClass
{

    public static Classifier[] model = new Classifier[]{
            new ZeroR(),
            new OneR(),

            //new Id3(),
            new J48(),
            new SimpleCart(),

            new NaiveBayes(),
            new NBTree(),

            new RandomTree(),
            new BayesNet(),

            new SMO(),
            new IBk()
    };


    public static Classifier[] classifiers = new Classifier[]{
            new ZeroR(),
            new OneR(),

            //new Id3(),
            new J48(),
            new SimpleCart(),

            new NaiveBayes(),
            new NBTree(),

            new RandomTree(),
            new BayesNet(),

            new SMO(),
            new IBk()
    };



    public static String[] namesClassifier = {
            "ZeroR", "OneR",
            "C4.5/J48", "CART/Gini",
            "Naive Bayes", "Naive Bayes Tree",
            "Random Tree", "Baysian Networks",
            "SVM", "kNN (k=1)"
    };

    public static Vector dimention(String path) throws Exception
    {
        /**
         * This method return : ROW x COLUMN
         */

        Instances data = new DataSource(path).getDataSet(); // Load data

        int row = data.numInstances();      // return : NUMBER of Record(s)
        int column = data.numAttributes();  // return : NUMBER of Feature(s)

        Vector MxN = new Vector<Integer>(); // Declated a Vector named MxN ( Array is NOT good here. )


        MxN.add(row);       // Append NUMBER of ROW into MxN Vector.
        MxN.add(column);    // Append NUMBER of COLUMN into MxN Vector.


        return MxN; /* Finally return : [ROW, COLUMN] */

    }


    public static Vector dimention(Instances data) throws Exception
    {
        /**
         * This method also return : ROW x COLUMN
         */

        int row = data.numInstances();      // return : NUMBER of Record(s)
        int column = data.numAttributes();  // return : NUMBER of Feature(s)

        Vector MxN = new Vector<Integer>(); // Declated a Vector named MxN ( Array is NOT good here. )

        MxN.add(row);       // Append number of row into MxN Vector.
        MxN.add(column);    // Append number of column into MxN Vector.

        return MxN; /* Finally return : [ROW, COLUMN] */
    }


    // # 01
    public static void useTrainingData(String trainDataPath, String evaluationFile) throws Exception
    {

        Formatter out = new Formatter(new File(evaluationFile));

        Instances train = new DataSource(trainDataPath).getDataSet();

        //System.out.println(train);

        train.setClassIndex(train.numAttributes() - 1);

        for (int i = 0; i < model.length; i++)
        {

            long begin = System.currentTimeMillis();       // Evaluation is start.

            Evaluation evaluation = new Evaluation(train);

            model[i].buildClassifier(train);

            evaluation.evaluateModel(model[i], train);

            long end = System.currentTimeMillis();         // Evaluation is end.

            double timeElapsed = (double) (end - begin) / 1000.0;

            out.format("Classifier : %s", namesClassifier[i]);

            out.format(" [Time elapsed : " + timeElapsed + " (s).]\n");

            out.format("1. Accuracy : %.6s", evaluation.pctCorrect());
            out.format(" %%\n");

            out.format("2. Error Rate : %.6s\n", evaluation.errorRate());
            out.format("3. Sensitivity (Recall) : %.6s\n", evaluation.weightedRecall());
            out.format("4. Specificity : %.6s\n", evaluation.weightedTrueNegativeRate());
            out.format("5. Precision : %.6s\n", evaluation.weightedPrecision());
            out.format("6. F-Score : %.6s\n", evaluation.weightedFMeasure());
            out.format("--------------------------------------------------------------------");
            out.format("\n");

            System.out.println(namesClassifier[i] + " is done." + " [Time elapsed : " + timeElapsed + " (s).]");

        }
        out.format("\n");
        out.close();

    }


    // # 02
    public static void useSuppliedTestingData(String trainDataPath, String testDataPath) throws Exception
    {

        Formatter out = new Formatter(new File("/home/rafsanjani/Desktop/Evaluation.txt"));


        Instances train = new DataSource(trainDataPath).getDataSet();

        Instances test = new DataSource(testDataPath).getDataSet();

        train.setClassIndex(train.numAttributes() - 1);
        test.setClassIndex(test.numAttributes() - 1);


        for (int i = 0; i < model.length; i++)
        {

            long begin = System.currentTimeMillis();       // Evaluation is start.

            Evaluation evaluation = new Evaluation(train);

            model[i].buildClassifier(train);

            evaluation.evaluateModel(model[i], test);

            long end = System.currentTimeMillis();         // Evaluation is end.

            double timeElapsed = (double) (end - begin) / 1000.0;

            out.format("Classifier : %s", namesClassifier[i]);

            out.format(" [Time elapsed : " + timeElapsed + " (s).]\n");

            out.format("1. Accuracy : %.6s", evaluation.pctCorrect());
            out.format(" %%\n");

            out.format("2. Error Rate : %.6s\n", evaluation.errorRate());
            out.format("3. Sensitivity (Recall) : %.6s\n", evaluation.weightedRecall());
            out.format("4. Specificity : %.6s\n", evaluation.weightedTrueNegativeRate());
            out.format("5. Precision : %.6s\n", evaluation.weightedPrecision());
            out.format("6. F-Score : %.6s\n", evaluation.weightedFMeasure());
            out.format("--------------------------------------------------------------------");
            out.format("\n");

            System.out.println(namesClassifier[i] + " is done." + " [Time elapsed : " + timeElapsed + " (s).]");

        }
        out.format("\n");
        out.close();


    }

    // # 03
    public static void useCrossValidation(String trainDataPath, String evaluationFile, int kFolds) throws Exception
    {

        Formatter out = new Formatter(new File(evaluationFile));

        Instances trainData = new DataSource(trainDataPath).getDataSet();

        trainData.setClassIndex(trainData.numAttributes() - 1);


        for (int i = 0; i < model.length; i++)
        {
            long begin = System.currentTimeMillis();        // Evaluation is start.

            Evaluation evaluation = new Evaluation(trainData);

            model[i].buildClassifier(trainData);

            evaluation.crossValidateModel(model[i], trainData, kFolds, new Random(1));

            long end = System.currentTimeMillis();         // Evaluation is end.

            double timeElapsed = (double) (end - begin) / 1000.0;

            out.format("Classifier : %s", namesClassifier[i]);

            out.format(" [Time elapsed : " + timeElapsed + " (s).]\n");

            out.format("1. Accuracy : %.6s", evaluation.pctCorrect());
            out.format(" %%\n");

            out.format("2. Error Rate : %.6s\n", evaluation.errorRate());
            out.format("3. Sensitivity (Recall) : %.6s\n", evaluation.weightedRecall());
            out.format("4. Specificity : %.6s\n", evaluation.weightedTrueNegativeRate());
            out.format("5. Precision : %.6s\n", evaluation.weightedPrecision());
            out.format("6. F-Score : %.6s\n", evaluation.weightedFMeasure());
            out.format("--------------------------------------------------------------------");
            out.format("\n");

            System.out.println(namesClassifier[i] + " is done." + " [Time elapsed : " + timeElapsed + " (s).]");

        }
        out.format("\n");
        out.close();


    }

    // # 04
    public static void useSplitingTrainingData(String trainDataPath, double n) throws Exception
    {
        Formatter out = new Formatter(new File("/home/rafsanjani/Desktop/Evaluation.txt"));

        n = n / 100.0; // Example : train 70% = 0.70 and test 30% = 0.30

        Instances data = new DataSource(trainDataPath).getDataSet();
        data.randomize(new Random());

        data.setClassIndex(data.numAttributes() - 1);

        int trainSize = (int) Math.round(data.numInstances() * n);
        int testSize = data.numInstances() - trainSize;

        Instances train = new Instances(data, 0, trainSize);
        // Start from index = 0, trainSize records will take.
        Instances test = new Instances(data, trainSize, testSize);
        // Start from index = trainSize, testSize records will take.


        for (int i = 0; i < model.length; i++)
        {

            long begin = System.currentTimeMillis();      // Evaluation is start.

            Evaluation evaluation = new Evaluation(train);

            model[i].buildClassifier(train);

            evaluation.evaluateModel(model[i], test);

            long end = System.currentTimeMillis();         // Evaluation is end.

            double timeElapsed = (double) (end - begin) / 1000.0;

            out.format("Classifier : %s", namesClassifier[i]);

            out.format(" [Time elapsed : " + timeElapsed + " (s).]\n");

            out.format("1. Accuracy : %.6s", evaluation.pctCorrect());
            out.format(" %%\n");

            out.format("2. Error Rate : %.6s\n", evaluation.errorRate());
            out.format("3. Sensitivity (Recall) : %.6s\n", evaluation.weightedRecall());
            out.format("4. Specificity : %.6s\n", evaluation.weightedTrueNegativeRate());
            out.format("5. Precision : %.6s\n", evaluation.weightedPrecision());
            out.format("6. F-Score : %.6s\n", evaluation.weightedFMeasure());
            out.format("--------------------------------------------------------------------");
            out.format("\n");

            System.out.println(namesClassifier[i] + " is done." + " [Time elapsed : " + timeElapsed + " (s).]");

        }
        out.format("\n");
        out.close();

    }

    public static void save(Instances data) throws IOException
    {
        ArffSaver file = new ArffSaver();
        file.setInstances(data);
        file.setFile(new File("/home/rafsanjani/Desktop/ClusterData.arff"));
        file.writeBatch();
    }

    public static void display(int records, int C)
    {
        System.out.println("\nNumber of Records is : " + records);
        System.out.println("Correclty Classified is : " + C + "\nMisclassified is : " + (records - C));

        System.out.printf("Accuracy is : %.2f", ((double) C / records) * 100.0);
        System.out.println(" %");
    }


    public static Instances removeAttribute(Instances data, String index) throws Exception
    {
        Remove remove = new Remove();
        String[] options = new String[]{"-R", index};

        remove.setOptions(options);
        remove.setInputFormat(data);
        data = Filter.useFilter(data, remove);

        return data;
    }


    public static void actual_VS_predicted(String trainDataPath) throws Exception
    {
        Instances train = new DataSource(trainDataPath).getDataSet();

        /**
         * If you to remove any attribute(s)
         * train = removeAttribute(train, "1-3");
         *
         */

        //train = removeAttribute(train, "1-3");

        train.setClassIndex(train.numAttributes() - 1);

        J48 model = new J48(); // Just One ...

        /**
         * -C = Confidence Factor
         * -M = minimum number of objects
         * "-C", "0.25", "-M", "30"
         *
         * (For kNN, "-K", "3")
         *
         */

        //String[] options = new String[]{"-C", "0.15", "-M", "5"};
        //model.setOptions(options);

        model.buildClassifier(train);

        int C = 0;
        for (int i = 0; i < train.numInstances(); i++)
        {

            Instance eachRecord = train.instance(i);

            String actual = train.classAttribute().value((int) train.instance(i).classValue());

            String predected = train.classAttribute().value((int) model.classifyInstance(eachRecord));

            System.out.print((i + 1) + " ");

            if (actual.equals(predected))
            {
                System.out.print("is correctly classified. ");
                C++;
            } else
            {
                System.out.print(" is misclassified. ");
            }
            System.out.println("[ " + "Actual: " + actual + ", Prediction: " + predected + " ]");


        }

        int records = train.numInstances();

        display(records, C);
    }


    public static void actual_VS_predicted(String trainDataPath, String testDataPath) throws Exception
    {
        Instances train = new DataSource(trainDataPath).getDataSet();

        Instances test = new DataSource(testDataPath).getDataSet();

        train.setClassIndex(train.numAttributes() - 1);
        test.setClassIndex(test.numAttributes() - 1);

        Classifier model = new J48(); // Just One ...
        model.buildClassifier(train);

        int C = 0;
        for (int i = 0; i < test.numInstances(); i++)
        {

            Instance eachRecord = test.instance(i);

            String actual = test.classAttribute().value((int) test.instance(i).classValue());

            String predected = test.classAttribute().value((int) model.classifyInstance(eachRecord));

            System.out.print((i + 1) + " ");

            if (actual.equals(predected))
            {
                System.out.print("is correctly classified. ");
                C++;
            } else
            {
                System.out.print(" is misclassified. ");
            }
            System.out.println("[ " + "Actual: " + actual + ", Prediction: " + predected + " ]");

        }

        int records = test.numInstances();
        display(records, C);

    }


    public static void launchClassifications() throws Exception
    {

        //String trainDataPath = "/home/rafsanjani/Installed/WEKA/weka-3-9-1/data/segment-challenge.arff";
        String trainDataPath = "/home/rafsanjani/MyDrive/Education/171/CSI_416/MD_RAFSAN_JANI_011_141_144/Data_Sets/Chronic_Kidney_Disease_full.arff";
        String testDataPath = "/home/rafsanjani/Installed/WEKA/weka-3-9-1/data/segment-test.arff";


        // Load "*.arff" to Instances ...
        Instances train = new DataSource(trainDataPath).getDataSet();
        Instances test = new DataSource(testDataPath).getDataSet();


//         // Calculate Dimension : row x column (Using : File Path):
//         System.out.println("Using : File Path :");
//         System.out.println("Train Data Dimension : "+new Dimension().dimention(trainDataPath));
//         System.out.println("Test Data Dimension : "+new Dimension().dimention(testDataPath)+"\n");


//         // Calculate Dimension : row x column (Using : Instances):
//         System.out.println("Using : Instances :");
//         System.out.println("Train Data Dimension : "+new Dimension().dimention(train));
//         System.out.println("Test Data Dimension : "+new Dimension().dimention(test)+"\n");


//         // Peek only row or column (As, it'll return a vector.)
//         System.out.println(new Dimension().dimention(trainDataPath).elementAt(0));
//         System.out.println(new Dimension().dimention(testDataPath).elementAt(1));


        /**
         * Run only one for evaluation.
         *
         */

//        useTrainingData(trainDataPath);
//        useSuppliedTestingData(trainDataPath, testDataPath);
//        useSplitingTrainingData(trainDataPath, 70); // n = trainData in %
        useCrossValidation(trainDataPath, "/home/rafsanjani/Chronic_Kidney_Disease.txt", 10);


        //////////////////////////////////////////////////////////////////////////////


        /**
         * See on eyes, how to compare and calculate actual and predicted classifier ?
         *
         * One parameter (trainDataPath)
         * actual_VS_predicted(trainDataPath);
         *
         * Two parameter (trainDataPath, testDataPath)
         * actual_VS_predicted(trainDataPath, testDataPath);
         *
         */


        //////////////////////////////////////////////////////////////////

        /**
         * Remove Feature one by one or a chunk ...
         */

//         Instances newData = removeAttribute(train, "1-15");

//         save(newData);


        System.out.println("\n" + "----------------------- Done ----------------------- :)");

    }

    public static void launchClusters()
    {
        System.out.println("Clusters !");
    }

    public static Instances removeClassValues(Instances data, String indices) throws Exception
    {

        RemoveWithValues remove = new RemoveWithValues();

        String[] options = new String[]{"-C", ("" + data.numAttributes()), "-L", indices};

        remove.setOptions(options);
        remove.setInputFormat(data);
        data = Filter.useFilter(data, remove);

        return data;

    }

    public static void debugClassification() throws Exception
    {
        // Load file as "*.arff" format from file path
        // Instances train = new DataSource("/home/rafsanjani/MyData/Education/Research/NSL_KDD/DATA_SETS/Given_Dr_DMF/nsl_kdd_train.arff").getDataSet();
        Instances train = new DataSource("/home/rafsanjani/Documents/Asif Mahbub - 011 121 089/Datasets/Chronic Kidney DIsease/chronic_kidney_disease_full.arff").getDataSet();

        // train = removeClassValues(train, "3,5,8,9,17,20,23");

        // Set the class Feature at index 4.
        train.setClassIndex(train.numAttributes() - 1);


        // Classifier name is J48/C4.5
        J48 model = new J48();

        // ( Declated a  Evaluate Class ) where "train" data will read.
        Evaluation evaluation = new Evaluation(train);


        // Build the model as J48/C4.5
        model.buildClassifier(train);

        // Evaluating the model : useOnlyTrainData
        // evaluation.evaluateModel(model, train);

        // Evaluating the model : kFold (Naturall, k=10) CrossValidation
        evaluation.crossValidateModel(model, train, 10, new Random(1));


        // Display what you want..
        System.out.println("Total Instances : " + (int) evaluation.numInstances());
        System.out.println("Total Features : " + (train.numAttributes() - 1) + "\n");

        System.out.printf("Accuracy : " + "%f\n", evaluation.pctCorrect());              // Accuracy
        System.out.printf("Error Rate : " + "%f\n", evaluation.errorRate());             // Error Rate
        System.out.printf("Recall : " + "%f\n", evaluation.weightedRecall());            // Recall (Weighted)
        System.out.printf("Precision : " + "%f\n", evaluation.weightedPrecision());       // Precision (Weighted)
        System.out.printf("F-Measure : " + "%f\n\n", evaluation.weightedFMeasure());
        System.out.printf("TP Rate : " + "%f\n", evaluation.weightedTruePositiveRate());
        System.out.printf("TN Rate : " + "%f\n", evaluation.weightedTrueNegativeRate());
        System.out.printf("FP Rate : " + "%f\n", evaluation.weightedFalsePositiveRate());
        System.out.printf("FN Rate : " + "%f\n\n", evaluation.weightedFalseNegativeRate());

        System.out.println(evaluation.toMatrixString());        // Confusion Matrix

//        System.out.println(model);

//        System.out.println(model.measureNumLeaves());
//        System.out.println(model.measureTreeSize());

        //System.out.println(model);

        System.out.println(evaluation.numTruePositives(0));
        System.out.println(evaluation.numFalsePositives(0));
        System.out.println(evaluation.numTrueNegatives(0));
        System.out.println(evaluation.numFalseNegatives(0));
        System.out.println("--------------------------------------");
        System.out.println(evaluation.numTruePositives(1));
        System.out.println(evaluation.numFalsePositives(1));
        System.out.println(evaluation.numTrueNegatives(1));
        System.out.println(evaluation.numFalseNegatives(1));
    }


    public static void debugCluster() throws Exception
    {
        // Before Clustering
        //useTrainingData("/home/rafsanjani/Installed/WEKA/weka-3-9-1/data/weather.numeric.arff", "/home/rafsanjani/Desktop/preEvaluation.txt");
        useCrossValidation("/home/rafsanjani/Installed/WEKA/weka-3-9-1/data/weather.numeric.arff", "/home/rafsanjani/Desktop/preEvaluation.txt", 10);

        Instances train = new DataSource("/home/rafsanjani/Installed/WEKA/weka-3-9-1/data/weather.numeric.arff").getDataSet();

        train = removeAttribute(train, "" + train.numAttributes());

        save(train);

        Instances noLebel = new DataSource("/home/rafsanjani/Desktop/ClusterData.arff").getStructure();


        Formatter out = new Formatter(new File("/home/rafsanjani/Desktop/noLebelCluster.arff"));
        out.format(noLebel.toString());

        SimpleKMeans model = new SimpleKMeans();

        model.setNumClusters(2);
        model.buildClusterer(train);

        ClusterEvaluation evaluation = new ClusterEvaluation();
        evaluation.evaluateClusterer(train);


        //Instances saved = new DataSource("/home/rafsanjani/Installed/WEKA/weka-3-9-1/data/weather.numeric.arff").getStructure();

//        System.out.println(saved);

//        ArffSaver saver = new ArffSaver();
//        String[] cmdArray3 = {"bash", "-c", "sed -i '$d' /home/rafsanjani/Desktop/noLebelCluster.arff"};
//        Runtime.getRuntime().exec(cmdArray3);


        out.format("@attribute cluster {");

        boolean ensure = true;
        for (int i = 1; i <= 2; i++)
        {
            if (ensure)
                out.format("cluster" + i);
            else
            {
                out.format(",cluster" + i);
            }
            ensure = false;
        }
        out.format("}" + "\n");

        out.format("@data" + "\n");
        for (int i = 0; i < train.numInstances(); i++)
        {
            Instance each = train.get(i);
            out.format(each + ",cluster" + (model.clusterInstance(each) + 1) + "\n");
        }

        out.close();


        File inputFile = new File("/home/rafsanjani/Desktop/noLebelCluster.arff");
        File tempFile = new File("/home/rafsanjani/Desktop/Clustered.arff");

        BufferedReader reader = new BufferedReader(new FileReader(inputFile));
        BufferedWriter writer = new BufferedWriter(new FileWriter(tempFile));

        String currentLine;

        ensure = true;
        while ((currentLine = reader.readLine()) != null)
        {

            String trimmedLine = currentLine.trim();

            if (ensure)
            {
                if (trimmedLine.equals("@data") || trimmedLine.equals("@DATA"))
                {
                    ensure = false;
                    continue;
                }
            }
            writer.write(currentLine + System.getProperty("line.separator"));


        }
        writer.close();
        reader.close();

        // After Clustering
        // useTrainingData("/home/rafsanjani/Desktop/Clustered.arff", "/home/rafsanjani/Desktop/postEvaluation.txt");
        useCrossValidation("/home/rafsanjani/Desktop/Clustered.arff", "/home/rafsanjani/Desktop/postEvaluation.txt", 10);

    }


    public static void debug() throws Exception
    {
        Instances train = new DataSource("/home/rafsanjani/MyDrive/Education/Research/NSL_KDD/DATA_SETS/Given_Dr_DMF/nsl_kdd_train.arff").getDataSet();

        // train = removeClassValues(train, "3,5,8,9,17,20,23");

        // Set the class Feature at index 4.
        train.setClassIndex(train.numAttributes() - 1);


        //Classifier name is J48/C4.5
        J48 model = new J48();

        // ( Declated a  Evaluate Class ) where "train" data will read.
//        Evaluation evaluation = new Evaluation(train);


        // Build the model as J48/C4.5
        model.buildClassifier(train);

        Evaluation evaluation = new Evaluation(train);

        // Evaluating the model : useOnlyTrainData
        evaluation.evaluateModel(model, train);

        // Evaluating the model : kFold (Naturall, k=10) CrossValidation
//        evaluation.crossValidateModel(model, train, 10, new Random());

//        System.out.println(train.attribute(0).name());

//        System.out.println(evaluation.graph());

    }

    public static void associationRule() throws Exception
    {
        String trainDataPath = "/home/rafsanjani/Installed/WEKA/weka-3-9-1/data/weather.nominal.arff";
        Instances train = new DataSource(trainDataPath).getDataSet();

//        System.out.println(train);

        Apriori model = new Apriori();
        model.buildAssociations(train);

        System.out.println(model);

    }

    public static void combineModel() throws Exception
    {
        //LOAD dataset
        String trainDataPath = "/home/rafsanjani/Installed/WEKA/weka-3-9-1/data/weather.nominal.arff";
        Instances train = new DataSource(trainDataPath).getDataSet();

        //SET class index
        train.setClassIndex(train.numAttributes() - 1);

//        System.out.println(train);


        // AdaBoost
//        AdaBoostM1 model = new AdaBoostM1();
//        model.setClassifier(new J48());
//        model.setNumIterations(1);
//        model.buildClassifier(train);
//        System.out.println(model);



        // Bagging
//        Bagging model = new Bagging();
//        model.setClassifier(new RandomTree());
//        model.setNumIterations(25);
//        model.buildClassifier(train);
//        System.out.println(model);



        // Stacking
//        Stacking model = new Stacking();
//        model.setClassifiers(classifiers);   // model.setMetaClassifier(new J48());
//        model.buildClassifier(train);
//        System.out.println(model);

        Vote model = new Vote();
        model.setClassifiers(classifiers);
        model.buildClassifier(train);
        System.out.println(model);

    }

    public static void main(String[] args) throws Exception
    {
        // launchClassifications();
        // launchClusters();

        // debugClassification();
        // debugCluster();

        // associationRule();
        combineModel();

        // debug();

    }
}


