document.addEventListener('DOMContentLoaded', function() {
    // Preloader
    const preloader = document.querySelector('.preloader');
    
    // Hide preloader after 2 seconds
    setTimeout(() => {
        preloader.classList.add('fade-out');
        setTimeout(() => {
            preloader.style.display = 'none';
        }, 500);
    }, 2000);
    
    // Initialize particles.js
    if (document.getElementById('particles-js')) {
        particlesJS('particles-js', {
            "particles": {
                "number": {
                    "value": 80,
                    "density": {
                        "enable": true,
                        "value_area": 800
                    }
                },
                "color": {
                    "value": "#6c5ce7"
                },
                "shape": {
                    "type": "circle",
                    "stroke": {
                        "width": 0,
                        "color": "#000000"
                    },
                    "polygon": {
                        "nb_sides": 5
                    }
                },
                "opacity": {
                    "value": 0.5,
                    "random": false,
                    "anim": {
                        "enable": false,
                        "speed": 1,
                        "opacity_min": 0.1,
                        "sync": false
                    }
                },
                "size": {
                    "value": 3,
                    "random": true,
                    "anim": {
                        "enable": false,
                        "speed": 40,
                        "size_min": 0.1,
                        "sync": false
                    }
                },
                "line_linked": {
                    "enable": true,
                    "distance": 150,
                    "color": "#6c5ce7",
                    "opacity": 0.4,
                    "width": 1
                },
                "move": {
                    "enable": true,
                    "speed": 2,
                    "direction": "none",
                    "random": false,
                    "straight": false,
                    "out_mode": "out",
                    "bounce": false,
                    "attract": {
                        "enable": false,
                        "rotateX": 600,
                        "rotateY": 1200
                    }
                }
            },
            "interactivity": {
                "detect_on": "canvas",
                "events": {
                    "onhover": {
                        "enable": true,
                        "mode": "grab"
                    },
                    "onclick": {
                        "enable": true,
                        "mode": "push"
                    },
                    "resize": true
                },
                "modes": {
                    "grab": {
                        "distance": 140,
                        "line_linked": {
                            "opacity": 1
                        }
                    },
                    "bubble": {
                        "distance": 400,
                        "size": 40,
                        "duration": 2,
                        "opacity": 8,
                        "speed": 3
                    },
                    "repulse": {
                        "distance": 200,
                        "duration": 0.4
                    },
                    "push": {
                        "particles_nb": 4
                    },
                    "remove": {
                        "particles_nb": 2
                    }
                }
            },
            "retina_detect": true
        });
    }
    
    // File Upload Handling
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const uploadProgress = document.querySelector('.upload-progress');
    const progressFill = document.querySelector('.progress-fill');
    const progressText = document.querySelector('.progress-text span');
    const resultsSection = document.getElementById('results');
    
    if (dropZone) {
        // Click on drop zone triggers file input
        dropZone.addEventListener('click', () => {
            fileInput.click();
        });
        
        // Handle drag over
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('drag-over');
        });
        
        // Handle drag leave
        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('drag-over');
        });
        
        // Handle drop
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('drag-over');
            
            if (e.dataTransfer.files.length) {
                fileInput.files = e.dataTransfer.files;
                handleFiles(fileInput.files);
            }
        });
        
        // Handle file input change
        fileInput.addEventListener('change', () => {
            if (fileInput.files.length) {
                handleFiles(fileInput.files);
            }
        });
        
        // Simulate file upload and processing
        function handleFiles(files) {
            console.log('Files selected:', files);
            
            // Show upload progress
            uploadProgress.classList.remove('hidden');
            
            // Simulate upload progress
            let progress = 0;
            const interval = setInterval(() => {
                progress += Math.random() * 10;
                if (progress >= 100) {
                    progress = 100;
                    clearInterval(interval);
                    
                    // Show results after a short delay
                    setTimeout(() => {
                        uploadProgress.classList.add('hidden');
                        resultsSection.classList.remove('hidden');
                        resultsSection.classList.add('fade-in');
                        
                        // Scroll to results
                        setTimeout(() => {
                            resultsSection.scrollIntoView({ behavior: 'smooth' });
                        }, 500);
                    }, 500);
                }
                
                progressFill.style.width = `${progress}%`;
                progressText.textContent = `${Math.round(progress)}%`;
            }, 200);
        }
    }
    
    // MRI Slice Slider
    const mriSlider = document.getElementById('mriSlice');
    if (mriSlider) {
        mriSlider.addEventListener('input', (e) => {
            // In a real app, this would change the MRI slice image
            console.log('Slice changed to:', e.target.value);
        });
    }
    
    // FAQ Accordion
    const faqItems = document.querySelectorAll('.faq-item');
    if (faqItems.length) {
        faqItems.forEach(item => {
            const question = item.querySelector('.faq-question');
            
            question.addEventListener('click', () => {
                // Close all other items
                faqItems.forEach(otherItem => {
                    if (otherItem !== item && otherItem.classList.contains('active')) {
                        otherItem.classList.remove('active');
                    }
                });
                
                // Toggle current item
                item.classList.toggle('active');
            });
        });
    }
    
    // FAQ Category Filter
    const categories = document.querySelectorAll('.category');
    const faqCategories = document.querySelectorAll('.faq-category');
    
    if (categories.length && faqCategories.length) {
        categories.forEach(category => {
            category.addEventListener('click', () => {
                // Update active category
                categories.forEach(c => c.classList.remove('active'));
                category.classList.add('active');
                
                // Show selected category
                const categoryName = category.getAttribute('data-category');
                faqCategories.forEach(faqCategory => {
                    if (faqCategory.classList.contains(categoryName)) {
                        faqCategory.classList.add('show');
                    } else {
                        faqCategory.classList.remove('show');
                    }
                });
            });
        });
    }
    
    // Testimonial Carousel
    const testimonialSlides = document.querySelectorAll('.testimonial-slide');
    const prevBtn = document.querySelector('.carousel-prev');
    const nextBtn = document.querySelector('.carousel-next');
    const dots = document.querySelectorAll('.dot');
    
    if (testimonialSlides.length) {
        let currentSlide = 0;
        
        function showSlide(index) {
            testimonialSlides.forEach(slide => slide.classList.remove('active'));
            dots.forEach(dot => dot.classList.remove('active'));
            
            testimonialSlides[index].classList.add('active');
            dots[index].classList.add('active');
            currentSlide = index;
        }
        
        // Next slide
        if (nextBtn) {
            nextBtn.addEventListener('click', () => {
                let nextIndex = (currentSlide + 1) % testimonialSlides.length;
                showSlide(nextIndex);
            });
        }
        
        // Previous slide
        if (prevBtn) {
            prevBtn.addEventListener('click', () => {
                let prevIndex = (currentSlide - 1 + testimonialSlides.length) % testimonialSlides.length;
                showSlide(prevIndex);
            });
        }
        
        // Dot navigation
        dots.forEach((dot, index) => {
            dot.addEventListener('click', () => {
                showSlide(index);
            });
        });
        
        // Auto-advance slides
        setInterval(() => {
            let nextIndex = (currentSlide + 1) % testimonialSlides.length;
            showSlide(nextIndex);
        }, 5000);
    }
    
    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;
            
            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                targetElement.scrollIntoView({
                    behavior: 'smooth'
                });
            }
        });
    });
    
    // Animate elements when they come into view
    const animateOnScroll = () => {
        const elements = document.querySelectorAll('.slide-up, .slide-down, .slide-left, .slide-right, .scale-in');
        
        elements.forEach(element => {
            const elementPosition = element.getBoundingClientRect().top;
            const windowHeight = window.innerHeight;
            
            if (elementPosition < windowHeight - 100) {
                element.classList.add('show');
            }
        });
    };
    
    window.addEventListener('scroll', animateOnScroll);
    animateOnScroll(); // Run once on page load
});

const browseBtn = document.querySelector(".btn-browse");
const fileInput = document.getElementById("fileInput");
const resultSection = document.getElementById("results");
const resultImage = document.querySelector(".mri-viewer .mri-image img");

browseBtn.addEventListener("click", (e) => {
    e.preventDefault();
    fileInput.click();
});

fileInput.addEventListener("change", function () {
    const file = this.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    document.querySelector(".upload-progress").classList.remove("hidden");

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        document.querySelector(".upload-progress").classList.add("hidden");

        if (data.error) {
            alert("Error: " + data.error);
            return;
        }

        resultSection.classList.remove("hidden");

        // ✅ Update prediction details
        document.querySelector(".confidence-fill").style.width = `${data.confidence}%`;
        document.querySelector(".confidence-meter span").innerText = `${data.confidence}% Confidence`;
        document.querySelector(".findings-list").innerHTML = `
            <li class="finding-item">
                <i class="fas fa-brain"></i>
                <div>
                    <h4>Condition: ${data.predicted_class}</h4>
                    <p>Confidence Score: ${data.confidence}%</p>
                </div>
                <span class="severity ${data.confidence > 85 ? 'high' : data.confidence > 50 ? 'medium' : 'low'}">${data.confidence > 85 ? 'High' : data.confidence > 50 ? 'Medium' : 'Low'}</span>
            </li>
        `;

        // ✅ Display uploaded image in the result section
        const uploadedImageURL = URL.createObjectURL(file);
        resultImage.src = uploadedImageURL;

        // Optional: Scroll to result
        setTimeout(() => {
            resultSection.scrollIntoView({ behavior: "smooth" });
        }, 300);
    })
    .catch(err => {
        console.error(err);
        alert("Failed to analyze the image.");
    });
});
