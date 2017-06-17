package mrzResearchArena;

// WEKA library lib.
import weka.classifiers.*;
import weka.classifiers.bayes.*;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.*;
import weka.classifiers.meta.*;
import weka.classifiers.rules.*;
import weka.classifiers.trees.*;
import weka.clusterers.ClusterEvaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

// Core Basic lib.
import java.io.*;
import java.lang.*;
import java.util.*;


public class MainClassV1
{

    public static void run() throws Exception
    {

        // Load the dataset
        Instances train = new DataSource("/home/rafsanjani/Desktop/PR_Class_04/iris.arff").getDataSet();


        // Set the class Feature at index 4.
        train.setClassIndex(train.numAttributes() - 1);


        // Model
        J48 model = new J48();
        // NaiveBayes model = new NaiveBayes();
        // IBk model = new IBk();
        // OneR model = new OneR();
        // SMO model = new SMO();
        // NBTree model = new NBTree();
        // RandomForest model = new RandomForest();
        

        // Build model
        model.buildClassifier(train);

        // Evaluation
        Evaluation evaluation = new Evaluation(train);
        evaluation.evaluateModel(model, train);


        System.out.println("---------------------------------------------------------");
        System.out.println("Accuracy : "+evaluation.pctCorrect()+" %");
        System.out.println("Error Rate : "+evaluation.pctIncorrect()+" %");
        System.out.println("Sensitivity (Recall) : "+evaluation.weightedRecall());
        System.out.println("Specificity : "+evaluation.weightedTrueNegativeRate());
        System.out.println("Precision : "+evaluation.weightedPrecision());
        System.out.println("F-Score : "+evaluation.weightedFMeasure()+"\n");
        System.out.println(evaluation.toMatrixString());
        System.out.println("---------------------------------------------------------");

    }


    public static void main(String[] args) throws Exception
    {
        run();
    }
}


