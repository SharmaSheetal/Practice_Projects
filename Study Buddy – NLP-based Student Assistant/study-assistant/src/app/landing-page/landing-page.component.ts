import { Component } from '@angular/core';
import { Router } from '@angular/router';
import { HttpClient, HttpClientModule } from '@angular/common/http';
import { CommonModule } from '@angular/common';
import { QuizService } from './quiz.service'; // Assuming QuizService is in the same directory
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-landing-page',
  imports: [CommonModule, HttpClientModule, FormsModule],
  standalone: true,
  templateUrl: './landing-page.component.html',
  styleUrls: ['./landing-page.component.css']
})
export class LandingPageComponent {
  selectedFiles: File[] = [];
  uploadMessage: string = '';
  loading: boolean = false; // Flag for loading spinner
  mcqData: any = null; // To hold MCQ data received from the API

  constructor(private router: Router, private http: HttpClient, private quizService: QuizService) {}

  // Handle file selection
  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      let filesArray = Array.from(input.files);

      // Filter only allowed file types
      const allowedTypes = ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/plain'];
      const validFiles = filesArray.filter(file => allowedTypes.includes(file.type));

      if (filesArray.length > 5) {
        this.uploadMessage = "You can only upload up to 5 files.";
        this.selectedFiles = [];
      } else if (validFiles.length !== filesArray.length) {
        this.uploadMessage = "Only PDF, DOCX, and TXT files are allowed.";
        this.selectedFiles = [];
      } else {
        this.selectedFiles = validFiles;
        this.uploadMessage = `Selected files: ${this.selectedFiles.map(file => file.name).join(', ')}`;
      }
    }
  }

  // Handle file upload and processing
  uploadFiles(): void {
    if (this.selectedFiles.length === 0) {
      this.uploadMessage = "No files selected!";
      return;
    }

    this.uploadMessage = "Uploading files... Please wait.";
    this.loading = true; // Start loading spinner

    const formData = new FormData();
    formData.append('file', this.selectedFiles[0]); // Only send the first file

    // Make POST request to the Flask backend
    this.http.post('http://localhost:5002/upload', formData).subscribe(
      (response: any) => {
        this.loading = false; // Stop loading spinner
        this.uploadMessage = "Upload successful!";
        console.log(response)
        this.mcqData = response.mcq_data; // Store the MCQ data
        console.log('Uploaded file:', response.file_path);
        console.log('MCQ Data:', this.mcqData);

        // Transform the Flask response to the quiz request format
        const quizRequest = this.quizService.transformFlaskResponseToQuizRequest(this.mcqData);

        // Call the API to create a quiz
        this.http.post('http://localhost:8080/api/v1/quiz/create-quiz', quizRequest).subscribe(
          (quizResponse) => {
            console.log('Quiz Created Successfully:', quizResponse);
            // Optionally navigate to another page after quiz creation
            this.router.navigate(['/summary-page']);
          },
          (error) => {
            console.error('Error creating quiz:', error);
            this.uploadMessage = "Failed to create quiz. Please try again.";
          }
        );
      },
      (error) => {
        this.loading = false; // Stop loading spinner
        this.uploadMessage = "Upload failed. Please try again.";
        console.error('Error uploading file:', error);
      }
    );
  }
}
