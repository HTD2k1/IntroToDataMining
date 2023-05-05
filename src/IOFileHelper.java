import weka.core.*;
import weka.classifiers.*;
import weka.core.converters.CSVSaver;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import java.io.File;
import java.security.InvalidParameterException;
import weka.core.SerializationHelper;


public class IOFileHelper {
    public static Instances loadData(String path) throws Exception {
        DataSource source = new DataSource(path);
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);;
        System.out.println("Successfully load data.....");
        return data;
    }

    public static void saveFile(Instances data, String fileName, String fileType) throws Exception {
        String fullFileName = fileName + "." + fileType;
        switch (fileType) {
            case "csv":
                saveAsCsv(data, fullFileName);
                break;
            case "arff":
                saveAsArff(data, fullFileName);
                break;
            default:
                throw new InvalidParameterException("Invalid or unsupported file types");
        }
    }

    public static void saveClassifier(Classifier classifier, String fileName) throws Exception {
        SerializationHelper.write("models/"+fileName, classifier);
    }
    
    public static Classifier loadClassifier(String fileName) throws Exception {
        return (Classifier) SerializationHelper.read(fileName);
    }
    
    private static void saveAsCsv(Instances data, String fullFileName) throws Exception {
        CSVSaver csvSaver = new CSVSaver();
        csvSaver.setInstances(data);
        csvSaver.setFile(new File(fullFileName));
        csvSaver.writeBatch();
    }
    
    private static void saveAsArff(Instances data, String fullFileName) throws Exception {
        ArffSaver arffSaver = new ArffSaver();
        arffSaver.setInstances(data);
        arffSaver.setFile(new File(fullFileName));
        arffSaver.writeBatch();
    }
}
