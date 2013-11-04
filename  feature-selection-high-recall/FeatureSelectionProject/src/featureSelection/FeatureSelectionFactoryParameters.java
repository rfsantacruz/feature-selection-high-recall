package featureSelection;

import weka.classifiers.AbstractClassifier;
import weka.core.Instances;

public class FeatureSelectionFactoryParameters {

	//number of features
	private int numberOfFeature;
	
	//base classifier for wrrapers
	private AbstractClassifier classifier;
	
	//data where the feature algorithm will work to get header information
	private Instances formatData;


	public FeatureSelectionFactoryParameters(int numberOfFeature,
			AbstractClassifier classifier, Instances formatData) {
		super();
		this.numberOfFeature = numberOfFeature;
		this.classifier = classifier;
		this.formatData = formatData;
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
	
	
}
