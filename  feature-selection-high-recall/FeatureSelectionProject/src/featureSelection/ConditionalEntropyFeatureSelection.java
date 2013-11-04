package featureSelection;

import JavaMI.Entropy;
import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.AttributeEvaluator;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;

public class ConditionalEntropyFeatureSelection extends ASEvaluation implements AttributeEvaluator{

	private Instances dataDiscretized;

	public ConditionalEntropyFeatureSelection(){
		super();
	}

	//used to initialize the Attribute selection algorithm
	@Override
	public void buildEvaluator(Instances data) throws Exception {

		//int classIndex = data.classIndex();
		//discretizing the data
		Discretize disTransform = new Discretize();
		disTransform.setUseBetterEncoding(true);
		disTransform.setInputFormat(data);
		this.dataDiscretized = Filter.useFilter(data, disTransform);
	}

	//calculate the mutual information for a given attribute
	@Override
	public double evaluateAttribute(int idx) throws Exception {
		double[] classColumn = this.dataDiscretized.attributeToDoubleArray(this.dataDiscretized.classIndex());
		double[] feature = this.dataDiscretized.attributeToDoubleArray(idx);
		
		//H(Y|X)
		return Entropy.calculateConditionalEntropy(classColumn, feature);

	}

	//make some processing in the attributes using the index of the attributes sorted
	@Override
	public int[] postProcess(int[] attributeSet) throws Exception {
		return super.postProcess(attributeSet);
	}
}
