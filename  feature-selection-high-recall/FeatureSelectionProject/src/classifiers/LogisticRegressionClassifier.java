package classifiers;

import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.util.Arrays;

import problems.ClassificationProblem;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.functions.LibLINEAR;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;


public class LogisticRegressionClassifier extends LibLINEAR{


	//constructor
	public LogisticRegressionClassifier(){
		super();
		this.setSVMType(new SelectedTag(0, LibLINEAR.TAGS_SVMTYPE));
	}

	@Override
	public void buildClassifier(Instances data) throws Exception {
		//set the liblinear to use l2 regularized logistic regression
		this.setSVMType(new SelectedTag(0, LibLINEAR.TAGS_SVMTYPE));
		//build the classify = train
		super.buildClassifier(data);
	}

	@Override
	public double classifyInstance(Instance newInstance) throws Exception {
		//set the liblinear to use l2 regularized logistic regression
		this.setSVMType(new SelectedTag(0, LibLINEAR.TAGS_SVMTYPE));
		//classify
		return super.classifyInstance(newInstance);

	}


	@Override
	public void setOptions(String[] options) throws Exception {
		//work around to avoid the api print trash in the console
		PrintStream console = System.out;
		System.setOut(new PrintStream(new OutputStream() {
			@Override public void write(int b) throws IOException {}
		}));

		super.setOptions(options);

		//set the liblinear to use l2 regularized logistic regression
		this.setSVMType(new SelectedTag(0, LibLINEAR.TAGS_SVMTYPE));

		//work around to avoid the api print trash in the console
		System.setOut(console);
	}


	//test the model implementation
	public static void main(String[] args) {

		try {
			ClassificationProblem cp = new ClassificationProblem("data/iris.data.arff");
			AbstractClassifier classifier = new LogisticRegressionClassifier();
			classifier.buildClassifier(cp.getData());
			for (Instance intance : cp.getData()) {
				System.out.print(classifier.classifyInstance(intance) + " , ");
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}




}
