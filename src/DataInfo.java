import weka.core.*;
import weka.core.converters.CSVSaver;
import weka.core.converters.ArffSaver;
import java.io.File;
import java.security.InvalidParameterException;

import javax.imageio.metadata.IIOInvalidTreeException;
public class DataInfo {
    public static void dataInfo(Instances data){
        System.out.println("=== DATASET OVERVIEW ===");
        
        // Print dataset summary
        System.out.println("Number of instances: " + data.numInstances());
        System.out.println("Number of attributes: " + data.numAttributes());
        System.out.println();

        // Print table header
        System.out.println("=== ATTRIBUTES INFO ===");
        System.out.printf("%-25s %-25s %-10s%n", "Attribute", "Attr Types", "Index");

        // Iterate through attributes and print as a table
        for (int i = 0; i < data.numAttributes(); i++) {
            Attribute attribute = data.attribute(i);
            String attributeName = attribute.name();
            String attributeType = Attribute.typeToString(attribute);
            int attributeIndex = attribute.index();
            System.out.printf("%-25s %-25s %-10d%n", attributeName, attributeType, attributeIndex);
        }
        System.out.println();
    } 

    // public static void classInfo(Instances data){
    //     data.getClass()
    // }
    
    public static void checkIfDatasetImBalanced(Instances data){

        System.out.println("=== CHECK IF DATASET IS IMBALANCED ===");
          // Set the index of the 'stroke' attribute
          int strokeIndex = data.attribute("stroke").index();
          data.setClassIndex(strokeIndex);
  
          // Calculate the percentage of patients who had a stroke and those who did not have a stroke
          int strokeCount = 0;
          int noStrokeCount = 0;
          for (Instance instance : data) {
              if (instance.classValue() == 1.0) {
                  strokeCount++;
              } else {
                  noStrokeCount++;
              }
          }
  
          double strokePercentage = (double) strokeCount / data.numInstances() * 100;
          double noStrokePercentage = (double) noStrokeCount / data.numInstances() * 100;
  
          // Show the results
          System.out.printf("  Percentage of patient had a stroke: %% %.2f --> (%d patient)%n", strokePercentage, strokeCount);
          System.out.printf("  Percentage of patient did not have a stroke: %% %.2f --> (%d patient)%n", noStrokePercentage, noStrokeCount);
    } 

    public static void checkMissingValues(Instances data){
        int[] missingValueCounts = new int[data.numAttributes()];
        for (int i = 0; i < data.numInstances(); i++) {
            for (int j = 0; j < data.numAttributes(); j++) {
                if (data.instance(i).isMissing(j)) {
                    missingValueCounts[j]++;
                }
            }
        }

        System.out.printf("%-25s %-25s %-25s%n", "Attributes", "Missing Values Numbers", "Missing Percentage");
        for (int i = 0; i < data.numAttributes(); i++) {
            Attribute attribute = data.attribute(i);
            String attributeName = attribute.name();
            int missingValues = missingValueCounts[i];
            double missingPercentage = (double) missingValues / data.numInstances() * 100;

            System.out.printf("%-25s %-25d %-25.2f%n", attributeName, missingValues, missingPercentage);
        }
    }

    public static void displayTop10Rows(Instances data){
        String formatColumns = "%-14s";
        System.out.println("=== DISPLAY TOP 10 ROWS ===");
        for (int i = 0; i < data.numAttributes(); i++) {
            Attribute attribute = data.attribute(i);
            System.out.printf(formatColumns, attribute.name());
        }
        System.out.println();

        int numRowsToDisplay = Math.min(10, data.numInstances());
        for (int i = 0; i < numRowsToDisplay; i++) {
            Instance instance = data.instance(i);
            for (int j = 0; j < data.numAttributes(); j++) {
                System.out.printf(formatColumns, instance.toString(j));
            }
            System.out.println();
        }
    }

    public static void overview(Instances data){
        DataInfo.dataInfo(data);
        DataInfo.displayTop10Rows(data);
        DataInfo.checkIfDatasetImBalanced(data);
        DataInfo.checkMissingValues(data);
    }

    public static void saveFile(Instances data,String fileName, String fileType) throws Exception {
        String fullFileName = fileName+"."+ fileType;
        switch(fileType){
            case "csv":
                CSVSaver csvSaver = new CSVSaver();
                csvSaver.setInstances(data);
                csvSaver.setFile(new File(fullFileName));
                csvSaver.writeBatch();
                break;
            case "arff":
                // Create an ArffSaver object
                ArffSaver arffSaver = new ArffSaver();
                arffSaver.setInstances(data);
                arffSaver.setFile(new File(fullFileName));
                arffSaver.writeBatch();
                break;

            default:
            throw new InvalidParameterException("Invalid or unsupported file types");
        }


    }
}