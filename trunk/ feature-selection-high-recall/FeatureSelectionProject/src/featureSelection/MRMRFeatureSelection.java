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
	private double[] relevance;
	private double[][] redundancy;
	
	//intialize the evaluator
	@Override
	public void buildEvaluator(Instances data) throws Exception {
		
		Discretize disTransform = new Discretize();
		disTransform.setUseBetterEncoding(true);
		disTransform.setInputFormat(data);
		this.dataDiscretized = Filter.useFilter(data, disTransform);
		
		//relevance precomputing mutual information feature-label
		this.relevance = new double[this.dataDiscretized.numAttributes() - 1];
		double[] classATT = this.dataDiscretized.attributeToDoubleArray(this.dataDiscretized.classIndex());
		for (int i = 0; i < this.relevance.length; i++) {
			double[] featureI = this.dataDiscretized.attributeToDoubleArray(i);
			relevance[i] = MutualInformation.calculateMutualInformation(classATT, featureI);
		}
		
		//redundancy precomputing mutual information feature-feature
		this.redundancy = new double[this.dataDiscretized.numAttributes() - 1][this.dataDiscretized.numAttributes() - 1];
		for (int i = 0; i < redundancy.length; i++) {
			double[] featureI = this.dataDiscretized.attributeToDoubleArray(i);
			for (int j = 0; j < redundancy[i].length; j++) {
				double[] featureJ = this.dataDiscretized.attributeToDoubleArray(j);
				redundancy[i][j] = MutualInformation.calculateMutualInformation(featureI,featureJ);
			}
		}
		
		

	}

	//analyse subsets of attributes
	@Override
	public double evaluateSubset(BitSet subSet) throws Exception {

		if(subSet.cardinality() == 0)
			return 0;
		
		double relevance = 0.0;
		double redundancy = 0.0;
		double[] classATT = this.dataDiscretized.attributeToDoubleArray(this.dataDiscretized.classIndex());
		
		for (int i = subSet.nextSetBit(0); i >= 0; i = subSet.nextSetBit(i + 1)) {
			relevance += this.relevance[i];
			for (int j = subSet.nextSetBit(0); j >= 0; j = subSet.nextSetBit(j + 1)) {
				if(i != j){
					redundancy += this.redundancy[i][j];
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
