package featureSelection;

import java.util.List;

import org.apache.commons.lang3.ArrayUtils;

import com.google.common.base.Joiner;
import com.google.common.collect.Collections2;
import com.google.common.primitives.Ints;

import weka.classifiers.AbstractClassifier;
import weka.core.Instances;

public class FeatureSelectionFactoryParameters {

	//number of features
	private int numberOfFeature;
	
	//base classifier for wrrapers
	private AbstractClassifier classifier;
	
	//data where the feature algorithm will work to get header information
	private Instances formatData;
	
	//Start set of attributes to optimize the search
	private int[] featureStartSet;


	public FeatureSelectionFactoryParameters(int numberOfFeature,
			AbstractClassifier classifier, Instances formatData) {
		this(numberOfFeature, classifier,formatData,null);
	}
	public FeatureSelectionFactoryParameters(int numberOfFeature,
			AbstractClassifier classifier, Instances formatData, int[] featureStartSet) {
		super();
		this.numberOfFeature = numberOfFeature;
		this.classifier = classifier;
		this.formatData = formatData;
		this.featureStartSet = featureStartSet;
	}

	//getter and setter 
	public Instances getFormatData() {
		return formatData;
	}

	public void setFormatData(Instances formatData) {
		this.formatData = formatData;
	}

	public int getNumberOfFeature() {
		return numberOfFeature;
	}

	public void setNumberOfFeature(int numberOfFeature) {
		this.numberOfFeature = numberOfFeature;
	}

	public AbstractClassifier getClassifier() {
		return classifier;
	}

	public void setClassifier(AbstractClassifier classifier) {
		this.classifier = classifier;
	}
	public int[] getFeatureStartSet() {
		return featureStartSet;
	}
	public void setFeatureStartSet(int[] featureStartSet) {
		this.featureStartSet = featureStartSet;
	}
	//this code is a workaround because ranges in weka are indexed by one not by zero
	public String featuresSelected2WekaRangeRepresentation(){
		String stringRep = "";
		
		if(this.featureStartSet != null && this.featureStartSet.length > 0){
			Integer[] valuesModified = new Integer[this.featureStartSet.length];
			for (int i = 0; i < featureStartSet.length; i++) valuesModified[i] = featureStartSet[i] + 1;
			stringRep = Joiner.on(", ").skipNulls().join(valuesModified);
		}
		
		return stringRep;
	}
	
	
}
