import java_cup.runtime.lr_parser;
import weka.core.*;
import weka.filters.*;
import weka.filters.unsupervised.attribute.*;

public class DataPreprocessing {
    public static Instances run(Instances data) throws Exception {    
        data = removeIdColumn(data);
        data = numericToNominalValues(data);
        data = processBmiAttribute(data);
        data = replaceMissingValues(data);
        data = convertStrokeAttributeToNominal(data);
        
        return data;
    }

    private static Instances removeIdColumn(Instances data) throws Exception {
        Remove removeFilter = new Remove();
        removeFilter.setAttributeIndices("1");
        removeFilter.setInputFormat(data);
        
        return Filter.useFilter(data, removeFilter);
    }

    private static Instances processBmiAttribute(Instances data) {

        // WEKA wrongly identifies "bmi" attr as STRING, so change it to NUMERIC 
        int bmiIndex = data.attribute("bmi").index();
        int tempBmiIndex = bmiIndex + 1;
        Attribute tempAttr = new Attribute("tempBmi");
        data.insertAttributeAt(tempAttr, tempBmiIndex);

        for (Instance instance : data) {
            if (instance.stringValue(bmiIndex).equalsIgnoreCase("N/A")) {
                instance.setMissing(bmiIndex);
            } else {
                double bmiValue = Double.parseDouble(instance.stringValue(bmiIndex));
                instance.setValue(tempBmiIndex, bmiValue);
            }
        }

        data.deleteAttributeAt(bmiIndex);    
        data.renameAttribute(data.attribute("tempBmi"), "bmi");

        return data;
    }

    private static Instances replaceMissingValues(Instances data) throws Exception {
        ReplaceMissingValues replaceFilter = new ReplaceMissingValues();
        replaceFilter.setInputFormat(data);
        
        return Filter.useFilter(data, replaceFilter);
    }

    private static Instances numericToNominalValues(Instances data) throws Exception{
     NumericToNominal filter = new NumericToNominal();
     filter.setInputFormat(data);
     filter.setAttributeIndices("3,4");
     return Filter.useFilter(data, filter);
    }
    

    private static Instances convertStrokeAttributeToNominal(Instances data) throws Exception {
        NumericToNominal filter = new NumericToNominal();
        filter.setAttributeIndices("last");
        filter.setInputFormat(data);
        Instances filteredData = Filter.useFilter(data, filter);
        filteredData.setClassIndex(filteredData.numAttributes() - 1);
        
        return filteredData;
    }
}
