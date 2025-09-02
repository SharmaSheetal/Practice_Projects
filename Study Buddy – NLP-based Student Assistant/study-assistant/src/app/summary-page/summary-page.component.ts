import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-summary-page',
  imports: [CommonModule],
  templateUrl: './summary-page.component.html',
  styleUrls: ['./summary-page.component.css']
})
export class SummaryPageComponent implements OnInit {
  uploadedFile: { name: string; size: number; type: string } | null = null;

  constructor(private router: Router) {}

  ngOnInit(): void {
    const fileData = localStorage.getItem('uploadedFile');
    if (fileData) {
      this.uploadedFile = JSON.parse(fileData);
    }
  }

  goBack(): void {
    this.router.navigate(['/']);
  }
}
