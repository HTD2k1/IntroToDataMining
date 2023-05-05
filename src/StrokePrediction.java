import weka.core.*;
import weka.filters.*;
import weka.filters.supervised.instance.SMOTE;
import weka.classifiers.*;
import weka.classifiers.functions.*;
import weka.classifiers.trees.*;
import java.util.ArrayList;
import java.util.Random;

public class StrokePrediction {
    public static Instances[] splitData(Instances data, double trainPercentage) {
        int trainSize = (int) Math.round(data.numInstances() * trainPercentage / 100);
        int testSize = data.numInstances() - trainSize;
        data.randomize(new Random(42)); // randomize data before splitting
        Instances train = new Instances(data, 0, trainSize);
        Instances test = new Instances(data, trainSize, testSize);
        return new Instances[]{train, test};
    }

    //Step 2
    public static ArrayList<Classifier> getClassifiers() {
        ArrayList<Classifier> classifiers = new ArrayList<>();
        classifiers.add(new Logistic());
        classifiers.add(new J48());
        //classifiers.add(new SMO());
        RandomForest rf = new RandomForest();
        classifiers.add(rf);
        return classifiers;
    }

    public static ArrayList<String> getClassifierNames() {
        ArrayList<String> classifierNames = new ArrayList<>();
        classifierNames.add("Logistic regression");
        classifierNames.add("J48");
        classifierNames.add("Random Forest");
        return classifierNames;
    }

    //Step 3
    public static Instances addSMOTEFilter(Instances data) throws Exception{
        // Create a SMOTE filter
        SMOTE smote = new SMOTE();
        smote.setInputFormat(data);

        // Set the percentage of SMOTE instances to create (default is 100%)
        int[] classCounts = DataInfo.countClasses(data, data.classIndex());
        smote.setPercentage((double) (classCounts[0] - classCounts[1])/ classCounts[1] * 100);
        //smote.setPercentage(100);

        // Set the number of nearest neighbors to consider (default is 5)
        smote.setNearestNeighbors(5);

        // Set the random seed for reproducibility (optional)
        smote.setRandomSeed(100);

        // Apply the SMOTE filter
        Instances balancedData = Filter.useFilter(data, smote);

        return balancedData;
    }

    //Step 4 
    public static void evaluateClassifiers(Instances data, ArrayList<Classifier> classifiers, ArrayList<String> classifierNames) throws Exception {

        System.out.println("=== MODEL EVALUATION ===");
        for (int i = 0; i < classifiers.size(); i++) {
            Classifier classifier = classifiers.get(i);
            // Measure model building and prediction time
            long startTime = System.currentTimeMillis();
            Evaluation evaluation = kFoldCrossValidation(classifier, data, 10, new Random(10),classifierNames.get(i));
            long endTime = System.currentTimeMillis();
            long totalTime = endTime - startTime;

            // Display model result
            int classIndex = data.classAttribute().indexOfValue("1");
            double auc = evaluation.areaUnderROC(classIndex);
            System.out.println(String.format("*** %s ***",classifierNames.get(i).toUpperCase()));
            System.out.println(String.format("** Area ROC: %.3f", auc));
            System.out.println("F1 Score: "+ evaluation.fMeasure(1));
            System.out.println(evaluation.toClassDetailsString());
            System.out.println(String.format("** Build and prediction time:%.3f s", (double) totalTime/1000));
            System.out.println(evaluation.toMatrixString());

            System.out.println(evaluation.toSummaryString());
            
            // Output binary file
            IOFileHelper.saveClassifier(classifier, classifierNames.get(i).toUpperCase() + ".model");

        }
    }

    public static Instances addWeight(Instances data){
                // Set the weights for instances based on class values
                for (int i = 0; i < data.numInstances(); i++) {
                    Instance instance = data.instance(i);
                    double classValue = instance.classValue();
        
                    if (classValue == 1.0) {
                        instance.setWeight(0.6);
                    } else if (classValue == 0.0) {
                        instance.setWeight(0.1);
                    }
                }
                return data;
    }
    // public static void evaluateClassifiers(Instances data, ArrayList<Classifier> classifiers, ArrayList<String> classifierNames) throws Exception {

    //     Instances[] splitData = splitData(data, 80);
    //     Instances trainData = splitData[0];
    //     Instances testData = splitData[1];
    //     trainData = addSMOTEFilter(trainData);
    //     trainData = addWeight(trainData);
    //     DataInfo.checkIfDatasetImbalanced(testData);
    //     System.out.println("=== MODEL EVALUATION ===");
    //     DataInfo.checkIfDatasetImbalanced(trainData);
    //     for (int i = 0; i < classifiers.size(); i++) {
    //         Classifier classifier = classifiers.get(i);
    //         // Measure model building and prediction time
    //         long startTime = System.currentTimeMillis();
    //         Evaluation evaluation = kFoldCrossValidation(classifier, trainData, 10, new Random(42),classifierNames.get(i));
    //         long endTime = System.currentTimeMillis();
    //         long totalTime = endTime - startTime;

    //         // Display model result
    //         double auc = evaluation.areaUnderROC(trainData.classAttribute().indexOfValue("1"));
    //         System.out.println(String.format("*** %s ***",classifierNames.get(i).toUpperCase()));
    //         System.out.println(String.format("** Area ROC: %.3f", auc));
    //         System.out.println("F1 Score: "+ evaluation.weightedFMeasure());
    //         System.out.println(String.format("** Build and prediction time:%.3f s", (double) totalTime/1000));
    //         System.out.println(evaluation.toSummaryString());

    //         // Final evaluation
    //         Evaluation finalEval = new Evaluation(data);
    //         finalEval.evaluateModel(classifier, testData);
    //         System.out.println("======> Final evaluation ");
    //         double finalAuc = finalEval.areaUnderROC(trainData.classAttribute().indexOfValue("1"));
    //         System.out.println(String.format("** Area ROC: %.3f", finalAuc));
    //         System.out.println("F1 Score: "+ finalEval.weightedFMeasure());
    //         System.out.println(finalEval.toClassDetailsString());
    //         System.out.println(evaluation.toSummaryString());

        

    //         // Output binary file
    //         IOFileHelper.saveClassifier(classifier, classifierNames.get(i).toUpperCase() + ".model");

    //     }
    // }

    public static Evaluation kFoldCrossValidation(Classifier classifier, Instances data, int numFolds, Random random, String classifierName) throws Exception {
        data.randomize(random);
        if (data.classAttribute().isNominal()) {
            data.stratify(numFolds);
        }
    
        Evaluation eval = new Evaluation(data);
        
        for (int n = 0; n < numFolds; n++) {
            Instances train = data.trainCV(numFolds, n, random);
            Instances test = data.testCV(numFolds, n);
            train = addSMOTEFilter(train);
            classifier.buildClassifier(train);
            eval.evaluateModel(classifier, test);
        }
        return eval;
    }

   public static void run() throws Exception {
        Instances data = IOFileHelper.loadData("./data/stroke-prediction-dataset/healthcare-dataset-stroke-data.csv");
        data = DataPreprocessing.run(data);
        DataInfo.overview(data);
        IOFileHelper.saveFile(data,"preproccessed_stroke_data", "csv");
        ArrayList<Classifier> classifiers = getClassifiers();
        ArrayList<String> classifierNames = getClassifierNames();
        evaluateClassifiers(data, classifiers, classifierNames);
    }
}
