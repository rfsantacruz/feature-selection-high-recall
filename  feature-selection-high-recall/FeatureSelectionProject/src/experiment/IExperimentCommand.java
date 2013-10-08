package experiment;

import java.util.List;

import problems.ClassificationProblem;

public interface IExperimentCommand {

	public List<ExperimentReport> execute(ClassificationProblem cp);
	
}
