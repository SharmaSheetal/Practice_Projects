import { ComponentFixture, TestBed } from '@angular/core/testing';
import { LandingPageComponent } from './landing-page.component';
import { By } from '@angular/platform-browser';

describe('LandingPageComponent', () => {
  let component: LandingPageComponent;
  let fixture: ComponentFixture<LandingPageComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [LandingPageComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(LandingPageComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create the component', () => {
    expect(component).toBeTruthy();
  });

  it('should display the heading', () => {
    const headingElement = fixture.debugElement.query(By.css('h1')).nativeElement;
    expect(headingElement.textContent).toContain('Welcome to the Student Upload Portal');
  });

  it('should display the paragraph', () => {
    const paragraphElement = fixture.debugElement.query(By.css('p')).nativeElement;
    expect(paragraphElement.textContent).toContain('Upload your study materials in PDF, DOCX, or TXT format.');
  });

  it('should disable upload button when no file is selected', () => {
    const buttonElement = fixture.debugElement.query(By.css('button')).nativeElement;
    expect(buttonElement.disabled).toBeTrue();
  });

  it('should enable upload button when valid files are selected', () => {
    const files = [
      new File(['dummy content'], 'file1.pdf', { type: 'application/pdf' }),
      new File(['dummy content'], 'file2.docx', { type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' }),
      new File(['dummy content'], 'file3.txt', { type: 'text/plain' })
    ];

    component.onFileSelected({ target: { files: files } } as unknown as Event);
    fixture.detectChanges();

    const buttonElement = fixture.debugElement.query(By.css('button')).nativeElement;
    expect(buttonElement.disabled).toBeFalse();
  });

  it('should display an error when selecting more than 5 files', () => {
    const files = [
      new File(['dummy content'], 'file1.pdf', { type: 'application/pdf' }),
      new File(['dummy content'], 'file2.docx', { type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' }),
      new File(['dummy content'], 'file3.txt', { type: 'text/plain' }),
      new File(['dummy content'], 'file4.pdf', { type: 'application/pdf' }),
      new File(['dummy content'], 'file5.docx', { type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' }),
      new File(['dummy content'], 'file6.txt', { type: 'text/plain' }) // Extra file
    ];

    component.onFileSelected({ target: { files: files } } as unknown as Event);
    fixture.detectChanges();

    expect(component.uploadMessage).toBe("You can only upload up to 5 files.");
    expect(component.selectedFiles.length).toBe(0);
  });

  it('should display an error when selecting an invalid file type', () => {
    const files = [
      new File(['dummy content'], 'valid.pdf', { type: 'application/pdf' }),
      new File(['dummy content'], 'invalid.jpg', { type: 'image/jpeg' }) // Invalid file
    ];

    component.onFileSelected({ target: { files: files } } as unknown as Event);
    fixture.detectChanges();

    expect(component.uploadMessage).toBe("Only PDF, DOCX, and TXT files are allowed.");
    expect(component.selectedFiles.length).toBe(0);
  });

  it('should display a success message after uploading valid files', (done) => {
    const files = [
      new File(['dummy content'], 'file1.pdf', { type: 'application/pdf' }),
      new File(['dummy content'], 'file2.docx', { type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' }),
      new File(['dummy content'], 'file3.txt', { type: 'text/plain' })
    ];

    component.selectedFiles = files;
    component.uploadFiles();
    fixture.detectChanges();

    setTimeout(() => {
      expect(component.uploadMessage).toBe('Upload successful!');
      done();
    }, 2000);
  });
});
