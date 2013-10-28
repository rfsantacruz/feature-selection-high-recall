package classifiers;

import java.util.Arrays;

import problems.ClassificationProblem;
import weka.classifiers.functions.LibLINEAR;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;

public class SVMLinearClassifier extends AbstractLinearClassifier {


	//constructor
	public SVMLinearClassifier(){
		model =  new LibLINEAR();
		this.getModel().setSVMType(new SelectedTag(1, LibLINEAR.TAGS_SVMTYPE));
		classifierName = "LinearSVM";
	}

	@Override
	public void buildClassifier(Instances data)
			throws Exception {

		//set the liblinear to use L2-loss support vector machines (dual)
		this.getModel().setSVMType(new SelectedTag(1, LibLINEAR.TAGS_SVMTYPE));
		
		//debug:
		//System.out.println(Arrays.toString(this.getModel().getOptions()));

		//build the classify (= train)
		this.model.buildClassifier(data);

	}
	
	@Override
	public double classifyInstance(Instance arg0) throws Exception {
		//set the liblinear to use L2-loss support vector machines (dual)
		this.getModel().setSVMType(new SelectedTag(1, LibLINEAR.TAGS_SVMTYPE));
		//classify the new instance
		return this.model.classifyInstance(arg0);
	}
	
	//reset to default options
	@Override
	public void resetClassifier() {
		this.model = new LibLINEAR();
		this.getModel().setSVMType(new SelectedTag(1, LibLINEAR.TAGS_SVMTYPE));
		
	}
	
	@Override
	public void setOptions(String[] options) throws Exception {
		super.setOptions(options);
		//set the liblinear to use l2 regularized logistic regression
		this.getModel().setSVMType(new SelectedTag(1, LibLINEAR.TAGS_SVMTYPE));
	}

	//get and setters with cast
	private LibLINEAR getModel() {
		return (LibLINEAR)this.model ;
	}
	public void setModel(LibLINEAR model) {
		this.model = model;
	}

	//test the clasifier implmentation
	public static void main(String[] args) {
		
		try {
			
			ClassificationProblem cp = new ClassificationProblem("data/iris.data.arff");
			AbstractLinearClassifier classifier = new SVMLinearClassifier();
			classifier.buildClassifier(cp.getData());
			System.out.println(Arrays.toString(classifier.classifyAll(cp.getData(), null)));

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	

	



}
