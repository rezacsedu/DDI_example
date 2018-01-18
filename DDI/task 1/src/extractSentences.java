/**
  * Author: Sachin Joshi
  * Task 1
  * Subtask 1
  */

import java.io.*;

class extractSentences
{
public static void main(String args[])
{
try
{
// Open the text file to be read and the text file to write the extracted attributes
BufferedReader br=new BufferedReader(new FileReader("joined.txt")); // This text file contains all the 572 documents corresponding to DrugBank data set
BufferedWriter br1=new BufferedWriter(new FileWriter("sentence", true)); // This is the text file where all the extracted sentences will be written into
String s;

// Read the entire text file line by line
while((s=br.readLine())!=null)
 {
	for(int i=0; i<s.length(); i++)
	 {
		for(int j=i+1; j<s.length(); j++)
		 {
			
			// The program searches for the text phrase "sentence id" 
			if(((s.substring(i,j)).compareTo("sentence id")==0))
			 {
				for(int k=j+2;; k++)
				 {
					if(s.charAt(k)=='=')
					 {
						
						for(int z=k+1;;z++)
						{
						  if(s.charAt(z)=='>') // Whenever the phrase "sentence id" is encountered,the complete sentence corresponding to it is extracted until ">" is encountered. ">" marks the end of the sentence 
							{
							String s1=s.substring(k+2,z-2);
							s1=s1+".";
							System.out.println(s1);
							br1.write(s1); // Extracted Sentence written to the text file 
							break;
							}
						}	
						br1.newLine(); 
						break; 
					 }
				 }
			 }
		 }
	 }
 }//end of while loop

 br.close(); //Close the text file being read 
br1.close(); //Close the text file being written

}//end of try block

//Catch any Input/Output Exceptions
catch(IOException e)
{
e.printStackTrace();
}

}//end of main
}//end of class

 