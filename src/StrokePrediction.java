import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.LDA;
import weka.classifiers.Classifier;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.StringToNominal;
import weka.filters.supervised.attribute.NominalToBinary;
import weka.core.converters.CSVSaver;
import weka.filters.unsupervised.attribute.NominalToString;
import java.util.ArrayList;
import java.util.Random;
import java.io.File;

public class StrokePrediction {

    public static Instances loadData(String path) throws Exception {
        DataSource source = new DataSource(path);
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);
        return data;
    }

    public static Instances preprocessData(Instances data) throws Exception {
        Remove removeFilter = new Remove();
        removeFilter.setAttributeIndices("1");
        removeFilter.setInputFormat(data);
        data = Filter.useFilter(data, removeFilter);
    
        // Convert the class attribute (stroke) to nominal
        NumericToNominal numericToNominalFilter = new NumericToNominal();
        numericToNominalFilter.setAttributeIndices(String.valueOf(data.classIndex() + 1));
        numericToNominalFilter.setInputFormat(data);
        data = Filter.useFilter(data, numericToNominalFilter);
    
        // String to Nominal
        StringToNominal filter = new StringToNominal();
        filter.setInputFormat(data);
        data = Filter.useFilter(data, filter);

        // Save preprocessed data to a CSV file
        CSVSaver saver = new CSVSaver();
        saver.setInstances(data);
        saver.setFile(new File("preprocessed_data.csv"));
        saver.writeBatch();   

        return data;
    }

    public static ArrayList<Classifier> getClassifiers() {
        ArrayList<Classifier> classifiers = new ArrayList<>();
        classifiers.add(new Logistic());
        classifiers.add(new LDA());
        classifiers.add(new SMO());
        return classifiers;
    }

    public static ArrayList<String> getClassifierNames() {
        ArrayList<String> classifierNames = new ArrayList<>();
        classifierNames.add("LR");
        classifierNames.add("LDA");
        classifierNames.add("SVM");
        return classifierNames;
    }

    public static void evaluateClassifiers(Instances data, ArrayList<Classifier> classifiers, ArrayList<String> classifierNames) throws Exception {
        for (int i = 0; i < classifiers.size(); i++) {
            Classifier classifier = classifiers.get(i);
            Evaluation evaluation = new Evaluation(data);
            evaluation.crossValidateModel(classifier, data, 10, new Random(42));
            double auc = evaluation.areaUnderROC(data.classAttribute().indexOfValue("1"));
            System.out.println(String.format(">%s %.3f", classifierNames.get(i), auc));
        }
    }

    public static void run() throws Exception {
        Instances data = loadData("./data/stroke-prediction-dataset/healthcare-dataset-stroke-data.csv");
        data = preprocessData(data);
        // ArrayList<Classifier> classifiers = getClassifiers();
        // ArrayList<String> classifierNames = getClassifierNames();
        // evaluateClassifiers(data, classifiers, classifierNames);
    }
}
