package experiment;

import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.DirectoryIteratorException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;

import problems.ClassificationProblem;
import utils.IConstants;

public class ExperimentExecutor {

	//execute a command per arrf problem in a jar data set
	public List<ExperimentReport> executeCommandInJAR(IExperimentCommand cmd, String dataSetPath){

		List<ExperimentReport> results = new ArrayList<ExperimentReport>();
		
		Path dir = Paths.get(dataSetPath);
		try {
			JarFile file = new JarFile(dir.toFile());
			Enumeration<JarEntry> e = file.entries();
			while(e.hasMoreElements()){
				JarEntry entry = e.nextElement();
				String ext = com.google.common.io.Files.getFileExtension(entry.getName());
				if(ext.equalsIgnoreCase(IConstants.ARFF_EXTENSION)){
					InputStreamReader inpr = new InputStreamReader(file.getInputStream(entry)) ;
					ClassificationProblem cp = new ClassificationProblem(entry.getName(),inpr);
					results.addAll(cmd.execute(cp));
				}

			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		return results;
	}
	
	//execute a command per arrf problem in the files data set
	public List<ExperimentReport> executeCommandInFiles(IExperimentCommand cmd, String dataSetPath){

		List<ExperimentReport> results = new ArrayList<ExperimentReport>();
		
		Path dir = Paths.get(dataSetPath);
		try (DirectoryStream<Path> stream = Files.newDirectoryStream(dir)) {
			for (Path file: stream) {
				String ext = com.google.common.io.Files.getFileExtension(file.toString());
				if(ext.equalsIgnoreCase(IConstants.ARFF_EXTENSION)){
					ClassificationProblem cp = new ClassificationProblem(file.toFile().getPath());
					List<ExperimentReport> report = cmd.execute(cp);
					if(report != null)
						results.addAll(report);
				}
			}
		} catch (IOException | DirectoryIteratorException x) {
			x.printStackTrace();
		}
		
		return results;
	}

}
