import weka.core.*;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SMO;
import weka.classifiers.trees.J48;
import weka.classifiers.Classifier;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.*;
import java.util.ArrayList;
import java.util.Random;
import weka.filters.supervised.instance.SMOTE;
import weka.classifiers.trees.RandomForest;

public class StrokePrediction {
    public static Instances loadData(String path) throws Exception {
        DataSource source = new DataSource(path);
        Instances data = source.getDataSet();
        System.out.println("Successfully load data.....");
        return data;
    }

    // Step 1 
    public static Instances preprocessData(Instances data) throws Exception {    
            // Remove the id column
            Remove removeFilter = new Remove();
            removeFilter.setAttributeIndices("1");
            removeFilter.setInputFormat(data);
            data = Filter.useFilter(data, removeFilter);
            
            //Mark missing values of attr "bmi"
            // WEKA wrongly identifies "bmi" attr as STRING, so change it to NUMERIC 
            Attribute tempAttr = new Attribute("tempBmi");
            int bmiIndex = data.attribute("bmi").index();
            int tempBmiIndex = bmiIndex + 1;


            data.insertAttributeAt(tempAttr, bmiIndex+1);
            for (Instance instance : data) {
                if (instance.stringValue(bmiIndex).equalsIgnoreCase("N/A")) {
                    instance.setMissing(bmiIndex);
                }
                else{
                    double bmiValue = Double.parseDouble(instance.stringValue(bmiIndex));
                    // Copy `bmi` value to `tempBmi` value 
                    instance.setValue(tempBmiIndex, bmiValue);
                }
            }
            data.deleteAttributeAt(bmiIndex);    
            // Have a look at the attributes
            data.renameAttribute(data.attribute("tempBmi"), "bmi");

            ReplaceMissingValues replaceFilter = new ReplaceMissingValues();
            replaceFilter.setInputFormat(data);
            data = Filter.useFilter(data, replaceFilter);

            // Add NumericToNominal filter to change target class `stroke`'s attribute type to NOMINAL
            NumericToNominal filter = new NumericToNominal();
            filter.setAttributeIndices("last");
            filter.setInputFormat(data);
            data = Filter.useFilter(data, filter);

            data.setClassIndex(data.numAttributes() - 1);
            DataInfo.overview(data);  

            return data;
    }


    //Step 2
    public static ArrayList<Classifier> getClassifiers() {
        ArrayList<Classifier> classifiers = new ArrayList<>();
        classifiers.add(new Logistic());
        classifiers.add(new J48());
        // classifiers.add(new LDA());
        classifiers.add(new SMO());
        classifiers.add(new RandomForest());
        return classifiers;
    }

    public static ArrayList<String> getClassifierNames() {
        ArrayList<String> classifierNames = new ArrayList<>();
        classifierNames.add("LR");
        classifierNames.add("J48");
        // classifierNames.add("LDA");
        classifierNames.add("SVM");
        classifierNames.add("Random Forest");
        return classifierNames;
    }

    //Step 3
    public static Instances addSMOTEFilter(Instances data) throws Exception{
        // Create a SMOTE filter
        SMOTE smote = new SMOTE();
        smote.setInputFormat(data);

        // Set the percentage of SMOTE instances to create (default is 100%)
        smote.setPercentage(100);

        // Set the number of nearest neighbors to consider (default is 5)
        smote.setNearestNeighbors(5);

        // Set the random seed for reproducibility (optional)
        smote.setRandomSeed(42);

        // Apply the SMOTE filter
        Instances balancedData = Filter.useFilter(data, smote);

        return balancedData;
    }

    //Step 4 
    public static void evaluateClassifiers(Instances data, ArrayList<Classifier> classifiers, ArrayList<String> classifierNames) throws Exception {

        System.out.println("=== MODEL EVALUATION ===");
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
        data = addSMOTEFilter(data);
        DataInfo.saveFile(data,"preproccessed_stroke_data", "arff");
        ArrayList<Classifier> classifiers = getClassifiers();
        ArrayList<String> classifierNames = getClassifierNames();
        evaluateClassifiers(data, classifiers, classifierNames);

    }
}
