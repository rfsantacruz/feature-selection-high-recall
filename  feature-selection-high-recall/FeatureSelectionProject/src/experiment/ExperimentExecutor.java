package experiment;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.DirectoryIteratorException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Enumeration;
import java.util.List;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.filefilter.FileFileFilter;
import org.apache.commons.io.filefilter.FileFilterUtils;
import org.apache.commons.io.filefilter.TrueFileFilter;

import problems.ClassificationProblem;
import utils.IConstants;

public class ExperimentExecutor {

	private static ExperimentExecutor instance;

	public static synchronized ExperimentExecutor getInstance() {
		if (instance == null) {
			instance = new ExperimentExecutor();
		}
		return instance;
	}

	private ExperimentExecutor(){}


	//execute a command per arrf problem in a jar data set
	public List<ExperimentReport> executeCommandInJAR(IExperimentCommand cmd, String jardataSetPath){

		List<ExperimentReport> results = new ArrayList<ExperimentReport>();

		Path jarfile = Paths.get(jardataSetPath);
		try {
			JarFile file = new JarFile(jarfile.toFile());
			Enumeration<JarEntry> e = file.entries();
			while(e.hasMoreElements()){
				JarEntry entry = e.nextElement();

				List<ExperimentReport> partialResult = this.executeExperimentJAR(cmd, file, entry);
				if(partialResult != null && partialResult.size() > 0)
					results.addAll(partialResult);

			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		return results;
	}

	//execute a command per arrf problem in the files data set
	public List<ExperimentReport> executeCommandInFiles(IExperimentCommand cmd, String dataSetSourcePath) {

		List<ExperimentReport> results = new ArrayList<ExperimentReport>();
		File dir = new File(dataSetSourcePath);
		if(dir.isDirectory()){
			Collection<File> files = FileUtils.listFiles(dir, TrueFileFilter.INSTANCE, TrueFileFilter.INSTANCE);
			for (File file: files) {
				List<ExperimentReport> partialResult = this.executeExperimentFile(cmd, file.getPath());
				if(partialResult != null && partialResult.size() > 0)
					results.addAll(partialResult);
			}
		}

		return results;

	}
	//execute a command in a arff file
	public ExperimentReport executeCommandInFile(IExperimentCommand cmd, String filePath) {

		List<ExperimentReport> results = this.executeExperimentFile(cmd, filePath);		
		return results != null && results.size() > 0 ? results.get(0) : null;
	}

	//private menbers//*************8

	//execute a experiment with problems in files
	private List<ExperimentReport> executeExperimentFile(IExperimentCommand iecmd,  String filePath) {

		List<ExperimentReport> results = new ArrayList<ExperimentReport>();
		try {
			File file = new File(filePath);
			if(file != null && file.exists()){
				String ext = com.google.common.io.Files.getFileExtension(file.getPath());
				if(ext.equalsIgnoreCase(IConstants.ARFF_EXTENSION)){
					ClassificationProblem cp = new ClassificationProblem(file.getPath());
					List<ExperimentReport> report = iecmd.execute(cp);
					if(report != null && report.size() > 0)
						results.addAll(report);
				}
			}
		} catch (IOException e) {
			e.printStackTrace();
		}

		return results;
	}

	//execute a experiment with problems in jar
	private List<ExperimentReport> executeExperimentJAR(IExperimentCommand iecmd,  JarFile file, JarEntry entry){

		List<ExperimentReport> results = new ArrayList<ExperimentReport>();
		try {
			if(entry != null){
				String ext = com.google.common.io.Files.getFileExtension(entry.getName());
				if(ext.equalsIgnoreCase(IConstants.ARFF_EXTENSION)){
					InputStreamReader inpr = new InputStreamReader(file.getInputStream(entry));
					ClassificationProblem cp = new ClassificationProblem(entry.getName(),inpr);
					List<ExperimentReport> report = iecmd.execute(cp);
					if(report != null && report.size() > 0)
						results.addAll(report);
				}
			}
		} catch (IOException e) {
			e.printStackTrace();
		}

		return results;
	}

}
