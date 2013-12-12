package featureSelection.Evaluators;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.SubsetEvaluator;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.ReplaceMissingWithUserConstant;

import com.google.common.base.Joiner;

public abstract class OurBaseFeatureSelectionEvaluator extends ASEvaluation implements SubsetEvaluator  {


	protected Instances dataBinarized;

	//probs[i][yd][xi] = P(y_i| xi, fi)
	protected double[][][] probs;

	//count event y=yd, xi = xd for fi
	protected double[][][] count_fiyx;

	//count event xi = xd for fi
	protected double[][] count_fix;

	//laplace smoth paramter
	protected double lspValue = 1;


	//intialize the evaluator
	@Override
	public void buildEvaluator(Instances data) throws Exception {


		//discretize the data set
		Discretize disTransform = new Discretize();
		disTransform.setUseBetterEncoding(true);
		disTransform.setInputFormat(data);
		this.dataBinarized = Filter.useFilter(data, disTransform);

		//dealling with missing values
		Set<Integer> attibutesToAddMissing = new HashSet<Integer>();
		for (int attIndex = 0; attIndex < this.dataBinarized.numAttributes() - 1 ; attIndex++) {
			double[] attributeValues = this.dataBinarized.attributeToDoubleArray(attIndex);
			for (int d = 0; d < this.dataBinarized.numInstances(); d++) {
				if(Double.isNaN(attributeValues[d])){
					attibutesToAddMissing.add(attIndex + 1);
					break;
				}
			}
		}

		ReplaceMissingWithUserConstant filter = new ReplaceMissingWithUserConstant();
		String t =  Joiner.on(", ").skipNulls().join(attibutesToAddMissing);
		filter.setAttributes(t);
		filter.setNominalStringReplacementValue("'?'");
		filter.setInputFormat(this.dataBinarized);
		this.dataBinarized = Filter.useFilter(dataBinarized, filter);


		//variables to help in the vector instatiations
		int numberOfAttributes = this.dataBinarized.numAttributes() - 1;
		int numberOfClasses = this.dataBinarized.numClasses();
		int multplier = numberOfClasses;

		//instatiation
		this.count_fiyx = new double[numberOfAttributes][numberOfClasses][];
		this.probs = new double [numberOfAttributes][numberOfClasses][];
		this.count_fix = new double[numberOfAttributes][];

		for (int fi = 0; fi < numberOfAttributes; fi++) {	
			int xvalue = this.dataBinarized.attribute(fi).numValues();
			this.count_fix[fi] = new double[xvalue]; 

			for(int yvalue = 0 ; yvalue < numberOfClasses; yvalue++){
				this.count_fiyx[fi][yvalue] = new double[xvalue];
				this.probs[fi][yvalue] = new double[xvalue]; 
			}
		}

		//counts
		for(int fi = 0; fi < this.count_fiyx.length; fi++){

			//Gets the value of all instances for the attribute f  
			double[]  fAttibuteValues = this.dataBinarized.attributeToDoubleArray(fi);
			double[] ydLabels = this.dataBinarized.attributeToDoubleArray(this.dataBinarized.classIndex());

			for (int d = 0; d < fAttibuteValues.length; d++ ) {
				int xdi = (int)fAttibuteValues[d];
				int yd = (int)ydLabels[d];

				this.count_fix[fi][xdi]++;
				this.count_fiyx[fi][yd][xdi]++;

			}
		}
		//probability computation
		for (int fi = 0; fi < this.probs.length; fi++) {
			for (int y = 0; y < this.probs[fi].length; y++) {
				for (int x = 0; x < this.probs[fi][y].length; x++) {
					this.probs[fi][y][x] = (this.count_fiyx[fi][y][x] + lspValue)/(this.count_fix[fi][x] + lspValue * multplier);
				}
			}
		}

		/*//debug - this properties have to be true unless you create another math
		for (int fi = 0; fi < this.probs.length; fi++) {

			for(int y1 = 0; y1 < this.probs[y1].length; y1++){
				for (int x = 0; x < this.probs[fi][y1].length; x++) {
					double debug = 0;

					for (int y = 0; y < this.probs[fi].length; y++) {
						debug+= this.probs[fi][y][x];
					}

					System.out.println("Have to be 1: " + debug);
				}
			}
		}*/	
	}
}