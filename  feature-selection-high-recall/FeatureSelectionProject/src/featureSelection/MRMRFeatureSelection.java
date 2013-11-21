package featureSelection;

import java.util.BitSet;

import JavaMI.MutualInformation;

import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.SubsetEvaluator;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;

public class MRMRFeatureSelection extends ASEvaluation implements SubsetEvaluator   {
	
	private Instances dataDiscretized;
	
	//intialize the evaluator
	@Override
	public void buildEvaluator(Instances data) throws Exception {
		
		Discretize disTransform = new Discretize();
		disTransform.setUseBetterEncoding(true);
		disTransform.setInputFormat(data);
		this.dataDiscretized = Filter.useFilter(data, disTransform);

	}

	//analyse subsets of attributes
	@Override
	public double evaluateSubset(BitSet subSet) throws Exception {

		if(subSet.cardinality() == 0)
			return 0;
		
		double relevance = 0.0;
		double redundancy = 0.0;
		double[] classATT = this.dataDiscretized.attributeToDoubleArray(this.dataDiscretized.classIndex());
		
		
		for (int i = 0; i < this.dataDiscretized.numAttributes(); i++) {
			if(subSet.get(i)){
				double[] featureI = this.dataDiscretized.attributeToDoubleArray(i);
				relevance += MutualInformation.calculateMutualInformation(classATT, featureI);
				
				for (int j = i + 1; j < this.dataDiscretized.numAttributes(); j++) {
					if(i != j && subSet.get(j)){
						double[] featureJ = this.dataDiscretized.attributeToDoubleArray(j);
						redundancy += MutualInformation.calculateMutualInformation(featureI,featureJ);
					}
				}
				
			}
		}
		
		relevance = relevance / subSet.cardinality(); 
		redundancy = redundancy / Math.pow(subSet.cardinality(), 2);
		
		return relevance - redundancy;
	}

	//make some processing in the attributes using the index of the attributes sorted
	@Override
	public int[] postProcess(int[] attributeSet) throws Exception {
		return super.postProcess(attributeSet);
	}

}
