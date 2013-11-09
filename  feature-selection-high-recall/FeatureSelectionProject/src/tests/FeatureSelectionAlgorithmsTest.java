package tests;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.io.IOException;
import java.text.DecimalFormat;

import org.junit.Test;

import problems.ClassificationProblem;
import weka.attributeSelection.AttributeSelection;
import JavaMI.Entropy;
import JavaMI.MutualInformation;
import classifiers.NaiveBayesClassifier;
import featureSelection.EFeatureSelectionAlgorithm;
import featureSelection.FeatureSelectionFactoryParameters;
import featureSelection.FeatureSelectionFilterFactory;


public class FeatureSelectionAlgorithmsTest {

	@Test
	public void testJavaMIAPI() {

		DecimalFormat  formatter = new DecimalFormat ("#0.00"); 
		double[] X = new double[]{1,1,2,2,3,3,3,4,4,4,5,5,5,5,6,6};//l
		double[] Y = new double[]{1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3};//v

		//MI(X,Y) = MI(Y,X) - log2 acording to the documentation
		double mi = MutualInformation.calculateMutualInformation(Y, X);
		//System.out.println(mi);
		assertTrue("MI(Y|X) = 1.25, therefore the result is wrong", Double.valueOf(formatter.format(mi)) == 1.25);

		//H(X|Y) - log2 acording to the documentation
		double ce = Entropy.calculateConditionalEntropy(X, Y);
		//System.out.println(ce);
		assertTrue("H(X|Y) = 1.28, therefore the result is wrong", Double.valueOf(formatter.format(ce)) == 1.28);

		//H(X) - log2
		double e = Entropy.calculateEntropy(X);
		//System.out.println(e);
		assertTrue("H(X) = 2.53, therefore the result is wrong", Double.valueOf(formatter.format(e)) == 2.53);

		//MI(Y,X) = MI(X,Y) = H(X) - H(X|Y)
		double mi2 = Double.valueOf(formatter.format(e - ce)) ;
		assertTrue("MI(Y,X) = MI(X,Y) = H(X) - H(X|Y), therefore the result is wrong", Double.valueOf(formatter.format(mi)) == mi2);

	}

	//TODO: test the results of mutual information feature selection. i may use a artificial data set
	@Test
	public void testInformationGainFeatureSelection() {
		String dataset = "./TestDataSets/heart-statlog.arff";
		try {
			ClassificationProblem cp = new ClassificationProblem(dataset);

			//filter parameters
			FeatureSelectionFactoryParameters parameter = 
					new FeatureSelectionFactoryParameters(5, null, cp.getData());

			AttributeSelection filter = FeatureSelectionFilterFactory.getInstance()
					.createFilter(EFeatureSelectionAlgorithm.INFORMATIONGAIN_RANK, parameter);

			//exeute
			filter.SelectAttributes(cp.getData());
			int[] idxs = filter.selectedAttributes();
			System.out.println(EFeatureSelectionAlgorithm.INFORMATIONGAIN_RANK.name());
			for (int id : idxs) {
				System.out.println(cp.getData().attribute(id));
			}



		} catch (IOException e) {
			fail("problems to read the data set: " + e.getMessage());
		} catch (Exception e) {
			fail("problems in the feature selection engine: " + e.getMessage());
		}

	}

	//TODO: test the results of conditional entropy feature selection. i may use a artificial data set
	@Test
	public void testConditionalEntropy() {
		String dataset = "./TestDataSets/heart-statlog.arff";
		try {
			ClassificationProblem cp = new ClassificationProblem(dataset);

			//filter parameters
			FeatureSelectionFactoryParameters parameter = 
					new FeatureSelectionFactoryParameters(5, null, cp.getData());

			//filter application
			AttributeSelection filter = FeatureSelectionFilterFactory.getInstance()
					.createFilter(EFeatureSelectionAlgorithm.CONDITIONAL_ENTROPY_RANK, parameter);

			//exeute
			filter.SelectAttributes(cp.getData());
			int[] idxs = filter.selectedAttributes();
			System.out.println(EFeatureSelectionAlgorithm.CONDITIONAL_ENTROPY_RANK.name());
			for (int id : idxs) {
				System.out.println(cp.getData().attribute(id));
			}

		} catch (IOException e) {
			fail("problems to read the data set: " + e.getMessage());
		} catch (Exception e) {
			fail("problems in the feature selection engine: " + e.getMessage());
		}

	}

	//TODO: test the results of gain ratio feature selection. i may use a artificial data set
	@Test
	public void testGainRationRank() {
		String dataset = "./TestDataSets/heart-statlog.arff";
		try {
			ClassificationProblem cp = new ClassificationProblem(dataset);

			//filter parameters
			FeatureSelectionFactoryParameters parameter = 
					new FeatureSelectionFactoryParameters(5, null, cp.getData());

			//filter application
			AttributeSelection filter = FeatureSelectionFilterFactory.getInstance()
					.createFilter(EFeatureSelectionAlgorithm.GAINRATIO_RANK, parameter);

			//exeute
			filter.SelectAttributes(cp.getData());
			int[] idxs = filter.selectedAttributes();
			System.out.println(EFeatureSelectionAlgorithm.GAINRATIO_RANK.name());
			for (int id : idxs) {
				System.out.println(cp.getData().attribute(id));
			}

		} catch (IOException e) {
			fail("problems to read the data set: " + e.getMessage());
		} catch (Exception e) {
			fail("problems in the feature selection engine: " + e.getMessage());
		}

	}

	//TODO: test the results of Symmetrical Uncert feature selection. i may use a artificial data set
	@Test
	public void SymmetricalUncertRank() {
		String dataset = "./TestDataSets/heart-statlog.arff";
		try {
			ClassificationProblem cp = new ClassificationProblem(dataset);

			//filter parameters
			FeatureSelectionFactoryParameters parameter = 
					new FeatureSelectionFactoryParameters(5, null, cp.getData());

			//filter application
			AttributeSelection filter = FeatureSelectionFilterFactory.getInstance()
					.createFilter(EFeatureSelectionAlgorithm.SYMMETRICAL_UNCERT_RANK, parameter);

			//exeute
			filter.SelectAttributes(cp.getData());
			int[] idxs = filter.selectedAttributes();
			System.out.println(EFeatureSelectionAlgorithm.SYMMETRICAL_UNCERT_RANK.name());
			for (int id : idxs) {
				System.out.println(cp.getData().attribute(id));
			}

		} catch (IOException e) {
			fail("problems to read the data set: " + e.getMessage());
		} catch (Exception e) {
			fail("problems in the feature selection engine: " + e.getMessage());
		}

	}

	//TODO: Confirm if the correlation-based feature selection will be used. and test its results, i may use a artificial data set
	@Test
	public void testCorrelationBasedRankFeatureSlection(){
		String dataset = "./TestDataSets/heart-statlog.arff";
		try {
			ClassificationProblem cp = new ClassificationProblem(dataset);

			//filter parameters
			FeatureSelectionFactoryParameters parameter = 
					new FeatureSelectionFactoryParameters(5, null, cp.getData());

			//filter application
			AttributeSelection filter = FeatureSelectionFilterFactory.getInstance()
					.createFilter(EFeatureSelectionAlgorithm.CORRELATION_BASED_RANK, parameter);

			//exeute
			filter.SelectAttributes(cp.getData());
			int[] idxs = filter.selectedAttributes();
			System.out.println(EFeatureSelectionAlgorithm.CORRELATION_BASED_SUBSET.name());
			for (int id : idxs) {
				System.out.println(cp.getData().attribute(id));
			}

		} catch (IOException e) {
			fail("problems to read the data set: " + e.getMessage());
		} catch (Exception e) {
			fail("problems in the feature selection engine: " + e.getMessage());
		}

	}

	//TODO: Confirm if the correlation-based feature selection will be used. and test its results, i may use a artificial data set
	@Test
	public void testCorrelationBasedSubsetFeatureSlection(){
		String dataset = "./TestDataSets/heart-statlog.arff";
		try {
			ClassificationProblem cp = new ClassificationProblem(dataset);

			//filter parameters
			FeatureSelectionFactoryParameters parameter = 
					new FeatureSelectionFactoryParameters(5, null, cp.getData());

			//filter application
			AttributeSelection filter = FeatureSelectionFilterFactory.getInstance()
					.createFilter(EFeatureSelectionAlgorithm.CORRELATION_BASED_SUBSET, parameter);

			//exeute
			filter.SelectAttributes(cp.getData());
			int[] idxs = filter.selectedAttributes();
			System.out.println(EFeatureSelectionAlgorithm.CORRELATION_BASED_SUBSET.name());
			for (int id : idxs) {
				System.out.println(cp.getData().attribute(id));
			}

		} catch (IOException e) {
			fail("problems to read the data set: " + e.getMessage());
		} catch (Exception e) {
			fail("problems in the feature selection engine: " + e.getMessage());
		}

	}

	//TODO: Confirm if the correlation-based feature selection will be used. and test its results, i may use a artificial data set
	@Test
	public void testMRMRFeatureSlection(){
		String dataset = "./TestDataSets/heart-statlog.arff";
		try {
			ClassificationProblem cp = new ClassificationProblem(dataset);

			//filter parameters
			FeatureSelectionFactoryParameters parameter = 
					new FeatureSelectionFactoryParameters(5, null, cp.getData());

			//filter application
			AttributeSelection filter = FeatureSelectionFilterFactory.getInstance()
					.createFilter(EFeatureSelectionAlgorithm.MRMR_MI_BASED_SUBSET, parameter);

			//exeute
			filter.SelectAttributes(cp.getData());
			System.out.println(EFeatureSelectionAlgorithm.MRMR_MI_BASED_SUBSET.name());
			int[] idxs = filter.selectedAttributes();
			for (int id : idxs) {
				System.out.println(cp.getData().attribute(id));
			}

		} catch (IOException e) {
			fail("problems to read the data set: " + e.getMessage());
		} catch (Exception e) {
			fail("problems in the feature selection engine: " + e.getMessage());
		}

	}


	//TODO: check the scott answer about the foward selection
	@Test
	public void testForwardFeatureSelectionAlgorithm(){

		String dataset = "./TestDataSets/heart-statlog.arff";
		try {
			ClassificationProblem cp = new ClassificationProblem(dataset);

			//filter parameters
			FeatureSelectionFactoryParameters parameter = 
					new FeatureSelectionFactoryParameters(5, new NaiveBayesClassifier(), cp.getData());

			//filter application
			AttributeSelection filter = FeatureSelectionFilterFactory.getInstance()
					.createFilter(EFeatureSelectionAlgorithm.FORWARD_SELECTION_WRAPPER, parameter);

			//exeute
			filter.SelectAttributes(cp.getData());
			int[] idxs = filter.selectedAttributes();
			System.out.println(EFeatureSelectionAlgorithm.FORWARD_SELECTION_WRAPPER.name());
			for (int id : idxs) {
				System.out.println(cp.getData().attribute(id));
			}


		} catch (IOException e) {
			fail("problems to read the data set: " + e.getMessage());
		} catch (Exception e) {
			fail("problems in the feature selection engine: " + e.getMessage());
		}
	}

	//TODO: check the scott answer about the foward selection
	@Test
	public void testBackwardFeatureSelectionAlgorithm(){

		String dataset = "./TestDataSets/heart-statlog.arff";
		try {
			ClassificationProblem cp = new ClassificationProblem(dataset);

			//filter parameters
			FeatureSelectionFactoryParameters parameter = 
					new FeatureSelectionFactoryParameters(5, new NaiveBayesClassifier(), cp.getData());

			//filter application
			AttributeSelection filter = FeatureSelectionFilterFactory.getInstance()
					.createFilter(EFeatureSelectionAlgorithm.BACKWARD_SELECTION_WRAPPER, parameter);

			//exeute
			filter.SelectAttributes(cp.getData());
			int[] idxs = filter.selectedAttributes();
			System.out.println(EFeatureSelectionAlgorithm.BACKWARD_SELECTION_WRAPPER.name());
			for (int id : idxs) {
				System.out.println(cp.getData().attribute(id));
			}

		} catch (IOException e) {
			fail("problems to read the data set: " + e.getMessage());
		} catch (Exception e) {
			fail("problems in the feature selection engine: " + e.getMessage());
		}
	}



}
