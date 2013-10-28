package classifiers;

import java.util.Arrays;

import problems.ClassificationProblem;
import weka.classifiers.functions.LibLINEAR;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;


public class LogisticRegressionClassifier extends AbstractLinearClassifier{


	//constructor
	public LogisticRegressionClassifier(){
		model =  new LibLINEAR();
		this.getModel().setSVMType(new SelectedTag(0, LibLINEAR.TAGS_SVMTYPE));
		classifierName = "LogisticRegression";
	}

	//get and setter cats to acctual model
	public LibLINEAR getModel() {
		return (LibLINEAR)model;
	}
	public void setModel(LibLINEAR model) {
		this.model = model;
	}

	@Override
	public void buildClassifier(Instances data) throws Exception {
		//set the liblinear to use l2 regularized logistic regression
		this.getModel().setSVMType(new SelectedTag(0, LibLINEAR.TAGS_SVMTYPE));
		//build the classify = train
		this.model.buildClassifier(data);
	}

	@Override
	public double classifyInstance(Instance newInstance) throws Exception {
		//set the liblinear to use l2 regularized logistic regression
		this.getModel().setSVMType(new SelectedTag(0, LibLINEAR.TAGS_SVMTYPE));
		//classify
		return model.classifyInstance(newInstance);

	}
	
	//reset to default options
	@Override
	public void resetClassifier() {
		this.model = new LibLINEAR();
		this.getModel().setSVMType(new SelectedTag(0, LibLINEAR.TAGS_SVMTYPE));
	}
	
	@Override
	public void setOptions(String[] options) throws Exception {
		super.setOptions(options);
		//set the liblinear to use l2 regularized logistic regression
		this.getModel().setSVMType(new SelectedTag(0, LibLINEAR.TAGS_SVMTYPE));
	}
	
	
	//test the model implementation
	public static void main(String[] args) {
		
		try {
			ClassificationProblem cp = new ClassificationProblem("data/iris.data.arff");
			AbstractLinearClassifier classifier = new LogisticRegressionClassifier();
			classifier.buildClassifier(cp.getData());
			System.out.println(Arrays.toString(classifier.classifyAll(cp.getData(), null)));

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	
	

}
