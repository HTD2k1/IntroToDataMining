import weka.core.*;
public class DataInfo {
    public static void dataInfo(Instances data) {
        System.out.println("=== DATASET OVERVIEW ===");
        printDatasetSummary(data);
        printAttributesInfo(data);
    } 

    private static void printDatasetSummary(Instances data) {
        System.out.println("Number of instances: " + data.numInstances());
        System.out.println("Number of attributes: " + data.numAttributes());
        System.out.println();
    }

    private static void printAttributesInfo(Instances data) {
        System.out.println("=== ATTRIBUTES INFO ===");
        System.out.printf("%-25s %-25s %-10s%n", "Attribute", "Attr Types", "Index");

        for (int i = 0; i < data.numAttributes(); i++) {
            Attribute attribute = data.attribute(i);
            System.out.printf("%-25s %-25s %-10d%n", attribute.name(), Attribute.typeToString(attribute), attribute.index());
        }
        System.out.println();
    }

    public static void checkIfDatasetImbalanced(Instances data) {
        System.out.println("=== CHECK IF DATASET IS IMBALANCED ===");
        int[] classCounts = countClasses(data, data.attribute("stroke").index());
        printClassPercentages(classCounts, data.numInstances());
    }

    private static int[] countClasses(Instances data, int classIndex) {
        int[] classCounts = new int[data.numClasses()];
        for (Instance instance : data) {
            classCounts[(int) instance.classValue()]++;
        }
        return classCounts;
    }

    private static void printClassPercentages(int[] classCounts, int totalInstances) {
        for (int i = 0; i < classCounts.length; i++) {
            double percentage = (double) classCounts[i] / totalInstances * 100;
            System.out.printf("  Percentage of class %d: %% %.2f --> (%d instances)%n", i, percentage, classCounts[i]);
        }
    }

    public static void checkMissingValues(Instances data) {
        int[] missingValueCounts = countMissingValues(data);
        printMissingValueCounts(data, missingValueCounts);
    }

    private static int[] countMissingValues(Instances data) {
        int[] missingValueCounts = new int[data.numAttributes()];
        for (Instance instance : data) {
            for (int i = 0; i < data.numAttributes(); i++) {
                if (instance.isMissing(i)) {
                    missingValueCounts[i]++;
                }
            }
        }
        return missingValueCounts;
    }

    private static void printMissingValueCounts(Instances data, int[] missingValueCounts) {
        System.out.printf("%-25s %-25s %-25s%n", "Attributes", "Missing Values Numbers", "Missing Percentage");
        for (int i = 0; i < data.numAttributes(); i++) {
            Attribute attribute = data.attribute(i);
            int missingValues = missingValueCounts[i];
            double missingPercentage = (double) missingValues / data.numInstances() * 100;
            System.out.printf("%-25s %-25d %-25.2f%n", attribute.name(), missingValues, missingPercentage);
        }
    }

    public static void displayTopNRows(Instances data, int n) {
        System.out.println("=== DISPLAY TOP " + n + " ROWS ===");
        printHeader(data);
        printRows(data, n);
   
    }

    private static void printHeader(Instances data) {
        String formatColumns = "%-14s";
        for (int i = 0; i < data.numAttributes(); i++) {
            System.out.printf(formatColumns, data.attribute(i).name());
        }
        System.out.println();
    }
    
    private static void printRows(Instances data, int numRowsToDisplay) {
        numRowsToDisplay = Math.min(numRowsToDisplay, data.numInstances());
        String formatColumns = "%-14s";
        for (int i = 0; i < numRowsToDisplay; i++) {
            Instance instance = data.instance(i);
            for (int j = 0; j < data.numAttributes(); j++) {
                System.out.printf(formatColumns, instance.toString(j));
            }
            System.out.println();
        }
    }
    
    public static void overview(Instances data) {
        dataInfo(data);
        displayTopNRows(data, 10);
        checkIfDatasetImbalanced(data);
        checkMissingValues(data);
    }
}    