import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;

public class Romeo
{
    public static void main(String[] args) throws Exception
    {
        Instances abc = new DataSource("/home/rafsanjani/Desktop/PR_Class_04/iris.arff").getDataSet();

        // System.out.println(abc);

        abc.setClassIndex(abc.numAttributes() - 1);

        NaiveBayes model = new NaiveBayes();

        model.buildClassifier(abc);

        // Evaluation
        Evaluation result = new Evaluation(abc);
        result.evaluateModel(model, abc);

        // Show Results
        System.out.println("Accuracy : "+result.pctCorrect()+" %");
        System.out.println("Error Rate : "+result.pctIncorrect()+" %");

    }
}


