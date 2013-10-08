package utils;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
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
import experiment.ExperimentReport;


public class Util {


	public static List<ClassificationProblem> readAllFilesARFFFromDirectory(String path){
		List<ClassificationProblem> problems = new ArrayList<ClassificationProblem>();

		Path dir = Paths.get(path);
		try (DirectoryStream<Path> stream = Files.newDirectoryStream(dir)) {
			for (Path file: stream) {
				String ext = com.google.common.io.Files.getFileExtension(file.toString());
				if(ext.equalsIgnoreCase(IConstants.ARFF_EXTENSION))
					problems.add(new ClassificationProblem(file.toFile().getPath()));
			}
		} catch (IOException | DirectoryIteratorException x) {
			x.printStackTrace();
		}

		System.out.println(problems.size() +" ARFF files were read");
		return problems;
	}

	public static List<ClassificationProblem> readAllFilesARFFFromJar(String path){
		List<ClassificationProblem> problems = new ArrayList<ClassificationProblem>();

		Path dir = Paths.get(path);
		try {
			JarFile file = new JarFile(dir.toFile());
			Enumeration<JarEntry> e = file.entries();
			while(e.hasMoreElements()){
				JarEntry entry = e.nextElement();
				String ext = com.google.common.io.Files.getFileExtension(entry.getName());
				if(ext.equalsIgnoreCase(IConstants.ARFF_EXTENSION)){
					InputStreamReader inpr = new InputStreamReader(file.getInputStream(entry)) ;
					problems.add(new ClassificationProblem(entry.getName(),inpr));
				}

			}
		} catch (IOException e) {
			e.printStackTrace();
		}

		System.out.println(problems.size() +" ARFF files were read");
		return problems;
	}

	public static void saveExperimentReportAsCSV(String filePath, List<ExperimentReport> reports, String sep){
		Path file = Paths.get(filePath);
		try(PrintWriter out = new PrintWriter(file.toFile())){
			for (ExperimentReport report : reports) {
				if(sep == null)
					out.println(report.toCSV());
				else
					out.println(report.toCSV(sep));
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
	
	public static void saveExperimentReportAsXML(String filePath, List<ExperimentReport> reports){
		Path file = Paths.get(filePath);
		try(PrintWriter out = new PrintWriter(file.toFile())){
			for (ExperimentReport report : reports) {
				out.println(report.toXML());
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}

}
