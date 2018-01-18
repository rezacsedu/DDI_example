/**
  * Author: Sachin Joshi
  * Task 1
  * Subtask 3
  * Training Data set
  */

import java.io.*;

class extractWordsTags
{
public static void main(String args[])
	{	
		try
		{
			// Open the text file to be read and the text file to write the extracted attributes
			BufferedReader br=new BufferedReader(new FileReader("Tagged.txt")); // This file contains the tagged sentences
			BufferedWriter br1=new BufferedWriter(new FileWriter("Words.txt", true)); // This file will contain the extracted words where each word occupies a single line 
			BufferedWriter br2=new BufferedWriter(new FileWriter("Tags.txt", true)); // This file contains the POS tag corresponding to each word. Each tag occupies a single line
			String s;  String s2,s1; int k;
			
			//Read the entire text file line by line
			while((s=br.readLine())!=null)
			{
				int i=0;
				for(int j=i+1;j<s.length()-4; j++)
				{
					
					if(s.charAt(j)=='_') // Whenever a "_" is encountered, the word before it is extracted
					{
						s1=s.substring(i,j);
						System.out.print(s1); // Prints the extracted word to the screen
						br1.write(s1); // Extracted word is written to the text file
						br1.newLine();
						for(k=j+1;;k++)
						{
							if(s.charAt(k)==32) // Whenever a space is encountered, the POS tag before it of the recently extracted word is extracted
							{
								s2=s.substring(j+1,k);
								System.out.println("\t\t"+s2); // Prints the extracted POS tag to the screen
								br2.write(s2); // Extracted POS tag is written to the text file
								br2.newLine();
								break;
							}
						}
						i=k+1;
						
						
					}
					
				}
			}//end of while loop
			br.close(); //Close the text file being read 
			br1.close(); //Close the text file being written
			br2.close();
		}//end of try block

		//Catch any Input/Output Exceptions
		catch(IOException e)
		{
			e.printStackTrace();
		}

	}//end of main
}//end of class

 