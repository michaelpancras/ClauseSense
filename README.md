## ClauseSense 📝⚖️

AI-Powered Legal Document Analyzer

ClauseSense is a machine learning project that classifies legal contract clauses into specific categories to assist with contract review and analysis.
It explores both traditional ML baselines and deep learning transformers (LegalBERT), providing a full pipeline from preprocessing to evaluation.

📂 Project Structure

Phase 2 – Baseline Models (Scikit-learn)

Preprocessing with TF-IDF

Trained Logistic Regression, Linear SVM, Random Forest, and Naive Bayes

Evaluated with accuracy, macro-F1, and cross-validation

Phase 3 – Transformer Models (PyTorch/HuggingFace)

Fine-tuned LegalBERT for clause classification

Added imbalance-aware training (class weights), scheduler, and early stopping

Achieved ~73% accuracy and ~0.55 macro-F1 on CUAD subset

📊 Results (Baseline vs Transformer)
Model	Macro F1	Accuracy	Notes
Logistic Regression	0.61	~0.62	Best classical model
Linear SVM	0.60	~0.61	Competitive baseline
Random Forest	0.51	~0.55	Slower, less accurate
Multinomial NB	0.40	~0.42	Weak performance
LegalBERT (Phase 3)	0.55	0.73	Strongest overall
🚀 Tech Stack

Languages/Frameworks: Python, PyTorch, Scikit-learn, HuggingFace Transformers

Data: CUAD (Contract Understanding Atticus Dataset)

Evaluation: Accuracy, Macro F1, Classification Reports, Confusion Matrices

Visualization: Seaborn, Matplotlib

📌 Example Usage
demo_text = "This Agreement shall terminate upon thirty (30) days prior written notice."
pred, conf = predict_clause(demo_text)
print(f"Prediction: {pred} (confidence={conf:.2f})")


Output:
Prediction: Termination For Convenience (confidence=0.78)

📈 Learning Outcomes

Built an end-to-end NLP pipeline from TF-IDF to transformers

Benchmarked classical ML vs deep learning for legal NLP tasks

Gained experience with model fine-tuning, evaluation, and interpretability

Prepared foundation for future phases (summarization, risk scoring, UI deployment)
