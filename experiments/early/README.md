# راهنمای آزمایش‌های اولیهٔ FLIDS، مراحل ۰۱ تا ۰۹

شماره‌ها در این راهنما همان شمارهٔ تاریخی نوت‌بوک‌ها و آرشیوها هستند. در نسخهٔ بازنگری‌شدهٔ مقاله، شماره‌های ۸ تا ۱۰ برای **بسته‌های اجرایی ۱۰ تا ۱۲** هم به کار رفته‌اند؛ برای بازتولید آن نتایج از [راهنمای اصلی مخزن](../../README.md) استفاده کنید.

| مرحله | پرسش و کد قابل‌خواندن | ورودی اصلی | خروجی |
| --- | --- | --- | --- |
| ۰۱ | [ارزیابی خانوادهٔ حملهٔ دیده‌نشده](FLIDS_Experiment01_Engine_v1_1.py) | CSV رسمی DNN از Edge-IIoTset | نتایج و پیش‌بینی‌ها به تفکیک خانواده |
| ۰۲ | [تعمیم به دیتاست مبدأ کنارگذاشته‌شده](FLIDS_Experiment02_Engine.py) | NF-UQ-NIDS-v2 | امتیازها و نتایج چهار holdout |
| ۰۳ | [انتقال آستانه، کالیبراسیون و حذف ویژگی](FLIDS_Experiment03_Engine.py) | دادهٔ NetFlow v2 با پروتکل مرحلهٔ ۰۲ | امتیازهای ذخیره‌شده و جدول‌های کالیبراسیون |
| ۰۴ | [اثر شیوع فرضی حمله و بودجهٔ هشدار](FLIDS_Experiment04_Engine.py) | پیش‌بینی‌های مرحلهٔ ۰۳ | جدول‌های precision، حجم هشدار و عبور از بودجه |
| ۰۵ | [ناهمگنی کلاینت‌ها و مشارکت جزئی](FLIDS_Experiment05_Engine.py) | NF-UQ-NIDS-v2 | مقایسهٔ سناریوهای تخصیص و مشارکت |
| ۰۶ | [تفکیک عوامل ناهمگنی](FLIDS_Experiment06_Controlled_Heterogeneity_Engine.py) | NF-UQ-NIDS-v2 | تفاوت‌های جفت‌شده با کنترل مختلط |
| ۰۷ | [ترکیب آماری نتایج](FLIDS_Experiment07_Statistical_Synthesis_Engine.py) | شش ZIP اصلی مراحل ۰۱ تا ۰۶ | FLIDS_Experiment07_Statistical_Synthesis_Results.zip |
| ۰۸ | [تنظیم FedOpt فقط با داده‌های مبدأ](../retuning/engine08_original.py) | NetFlow v2، تقسیم‌ها و اثرانگشت‌ها | FLIDS_Experiment08_Results.zip |
| ۰۹ | [تنظیم مجدد FedProx](../retuning/engine09_retune.py) و [بررسی آستانه‌ها](../retuning/engine09_main.py) | نتایج ۰۸ و همان نمونه‌ها و تقسیم‌ها | FLIDS_Experiment09_Results.zip |

نام ZIP مراحل ۰۱ تا ۰۶ از الگوی `FLIDS_ExperimentNN_Results.zip` پیروی می‌کند. [ممیزی دیتاست و تقسیم‌ها](FLIDS_DatasetSplit_Audit_Engine.py) و [مدل‌های مقایسه‌ای اضافه برای ۰۱ و ۰۲](FLIDS_Master_Extra_Baselines.py) مسیرهای جداگانه‌ای در [اجراکنندهٔ اصلی](FLIDS_Master_Orchestrator.py) دارند.

## اجرا در Kaggle

- **۰۱ تا ۰۷:** [نوت‌بوک مادر](../../notebooks/legacy/FLIDS_Master_Experiments01_to07_ExpandedBaselines_Kaggle.ipynb) را import کنید. برای اجرای کامل هر دو CSV مربوط به Edge و NetFlow را متصل کنید. در سلول اول، `RUN_STAGES = None` کل مسیر را اجرا می‌کند. برای اجرای انتخابی، مثلاً `RUN_STAGES = ['03', '04']` تعیین کنید. برای ۰۷ به‌تنهایی، خروجی شش مرحلهٔ قبلی را متصل کنید.
- **۰۸ و ۰۹ از ابتدا:** [نوت‌بوک مشترک](../../notebooks/legacy/FLIDS_Experiment08_09_Full_Kaggle_AutoRun.ipynb) را با دیتاست ترکیبی NF-UQ-NIDS-v2 یا چهار CSV سازگار اجرا کنید. نوت‌بوک، آماده‌سازی، تنظیم اولیه، retune و ممیزی را به‌ترتیب انجام می‌دهد. برای ادامه از خروجی موجود، تنظیمات cache و resume را در همان نوت‌بوک ببینید.
- **۰۹ با ورودی‌های آماده:** [نوت‌بوک مستقل ۰۹](../../notebooks/legacy/FLIDS_Experiment09_Kaggle_AutoRun.ipynb) مسیر جایگزین است. بخش ورودی‌های نوت‌بوک مشخص می‌کند کدام آرشیو و دادهٔ آماده باید متصل باشد.

## بررسی و اجرای محلی

از ریشهٔ مخزن با محیط پایتونی که وابستگی‌های نوت‌بوک را دارد:

```bash
python checks/early_sources.py
python experiments/early/FLIDS_Experiment07_Statistical_Synthesis_Engine.py \
  --input-root /path/to/six-result-archives --output-root /path/to/results
```

برای آزمون نرم‌افزاری کوتاه ۰۸ و ۰۹، فایل‌های موتور باید در پوشهٔ کار کنار یکدیگر باشند. یک پوشهٔ تازه بسازید تا نتایج آزمایشی با نتایج پژوهشی مخلوط نشوند:

```bash
workdir=$(mktemp -d)
cp experiments/retuning/*.py "$workdir/"
python "$workdir/engine09_main.py" --work "$workdir" --smoke
```

فایل‌های پایتون این مسیر از سورس تعبیه‌شده در نوت‌بوک‌های اصلی استخراج شده‌اند. دستور بررسی بالا هر ۱۴ ماژول را با محتوای نوت‌بوک متناظر، بایت‌به‌بایت مقایسه می‌کند. دیتاست‌های خام و خروجی‌های حجیم جداگانه تهیه می‌شوند.
