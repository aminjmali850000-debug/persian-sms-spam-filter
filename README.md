# Persian SMS Spam Filter

Final project for the Building AI course

## Summary

A small tool that reads a Persian SMS and estimates the probability that it is spam. It uses a bag of words and a Naive Bayes classifier, so it is fast, needs no libraries and every decision can be explained.

## Background

Which problems does your idea solve?
* Many people receive spam SMS every day: fake prizes, loan offers, "your bank account is blocked" scams
* Some of these messages lead to fraud and stolen bank card details
* Most spam filters are built for English and work poorly for Persian text

My personal motivation: I receive these messages myself, and so do my family members, who are the ones most likely to click the links.

## How is it used?

The user gives a message and gets a spam probability. A message is flagged only if the probability is above 90%, because wrongly hiding a normal message is worse than letting some spam through.

```
python spam_filter.py                      # test the model on the data (see Full code below)
python spam_filter.py "برنده جایزه شدید لینک کلیک کنید"
# Spam probability: 100.0% -> SPAM
python spam_filter.py "سلام فردا جلسه داریم"
# Spam probability: 0.3% -> looks normal
```

Intended use: a phone app that moves suspicious SMS into a separate folder, so the user can still review them.

How it works, step by step:
1. The message is split into words (word order is thrown away)
2. P(spam | words) is computed with Bayes rule: prior P(spam) times the probability of each word given spam, compared with the same for normal messages
3. Words are assumed independent (the "naive" part) and Laplace smoothing handles words seen only in one class
4. The message is flagged if P(spam | words) > 0.9

## Data sources and AI methods

Data: `data/sms.csv` has 40 short Persian example messages (20 spam, 20 normal) that I wrote myself, so there are no privacy or copyright problems. This is only a demo set. A real version needs thousands of real labeled messages, collected with the permission of the people who sent and received them.

Methods:
* Bag of words
* Naive Bayes classifier (Bayes rule), written from scratch in Python
* Leave-one-out cross-validation to estimate how well it works on unseen messages (the method from the course)

Result on the demo data: 40 of 40 messages classified correctly with leave-one-out testing (0 normal messages flagged). This number is optimistic because the messages are short, clean and written by me. It shows that the method works, not that a real-world filter would be 100% accurate.

## Challenges

* The model does not understand meaning, only word statistics, so it cannot detect sarcasm or new scam styles
* Spammers change their wording and use Finglish (Persian in Latin letters), so the model needs regular retraining
* Normal messages (for example a real bank notification) can be wrongly flagged, so the user must be able to review flagged messages
* Persian text has spelling variants and half-spaces that a better tokenizer should normalize
* Privacy: SMS are personal, so the processing should happen on the device and messages should never be uploaded

## What next?

* Collect a much larger real dataset and test on messages the model has never seen
* Add TF-IDF to reduce the weight of very common words
* Compare with logistic regression or a small neural network
* Build a mobile app version
* Support Finglish and English

## Full code

Save this as `spam_filter.py`:

```python
"""Persian SMS spam filter: bag of words + Naive Bayes (no libraries needed)."""
import csv
import math
import random
import re
import sys
from collections import Counter


THRESHOLD = 0.9  # flag as spam only if we are quite sure


def tokenize(text):
    """Split a message into lowercase words (bag of words keeps no order)."""
    return re.findall(r"\w+", text.lower())


class NaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Laplace smoothing
        self.word_counts = {"spam": Counter(), "ham": Counter()}
        self.doc_counts = Counter()
        self.vocab = set()

    def fit(self, messages):
        for label, text in messages:
            words = tokenize(text)
            self.word_counts[label].update(words)
            self.doc_counts[label] += 1
            self.vocab.update(words)

    def spam_probability(self, text):
        """P(spam | words) using Bayes rule with the naive independence assumption."""
        total_docs = sum(self.doc_counts.values())
        log_p = {}
        for label in ("spam", "ham"):
            lp = math.log(self.doc_counts[label] / total_docs)  # prior
            total_words = sum(self.word_counts[label].values())
            for w in tokenize(text):
                if w not in self.vocab:
                    continue  # ignore words never seen in training
                count = self.word_counts[label][w]
                lp += math.log((count + self.alpha) / (total_words + self.alpha * len(self.vocab)))
            log_p[label] = lp
        m = max(log_p.values())
        e = {k: math.exp(v - m) for k, v in log_p.items()}
        return e["spam"] / (e["spam"] + e["ham"])

    def top_spam_words(self, n=8):
        """Words most typical for spam (ratio of spam to ham frequency)."""
        scores = []
        for w in self.vocab:
            s = self.word_counts["spam"][w] + self.alpha
            h = self.word_counts["ham"][w] + self.alpha
            if self.word_counts["spam"][w] >= 2:
                scores.append((s / h, w))
        return [w for _, w in sorted(scores, reverse=True)[:n]]


def load(path="data/sms.csv"):
    with open(path, encoding="utf-8") as f:
        return [(r["label"], r["text"]) for r in csv.DictReader(f)]


def evaluate(data):
    """Leave-one-out cross-validation (as in the course): train on all messages
    except one, test on the left-out message, repeat for every message."""
    correct = tp = fp = fn = 0
    for i, (label, text) in enumerate(data):
        model = NaiveBayes()
        model.fit(data[:i] + data[i + 1:])
        pred = "spam" if model.spam_probability(text) > THRESHOLD else "ham"
        correct += pred == label
        tp += pred == "spam" and label == "spam"
        fp += pred == "spam" and label == "ham"
        fn += pred == "ham" and label == "spam"
    print(f"Messages tested: {len(data)} (leave-one-out)")
    print(f"Accuracy: {correct / len(data):.0%}")
    print(f"Spam caught: {tp}, missed: {fn}, normal messages wrongly flagged: {fp}")


if __name__ == "__main__":
    data = load()
    if len(sys.argv) > 1:
        model = NaiveBayes()
        model.fit(data)
        p = model.spam_probability(" ".join(sys.argv[1:]))
        print(f"Spam probability: {p:.1%} -> {'SPAM' if p > THRESHOLD else 'looks normal'}")
    else:
        evaluate(data)
        model = NaiveBayes()
        model.fit(data)
        print("Typical spam words:", " ".join(model.top_spam_words()))
```

## Data

Save this as `data/sms.csv` (UTF-8):

```
label,text
spam,تبریک شما برنده یک میلیارد تومان جایزه شدید برای دریافت روی لینک کلیک کنید
spam,حساب بانکی شما مسدود شد برای فعال سازی فوری وارد لینک زیر شوید
spam,وام فوری بدون ضامن تا ۵۰۰ میلیون تومان همین حالا ثبت نام کنید
spam,شما برنده قرعه کشی ایرانسل شدید برای دریافت جایزه کلیک کنید
spam,کد تخفیف ۹۰ درصدی فقط تا امشب برای خرید وارد لینک شوید
spam,درآمد میلیونی در خانه بدون سرمایه همین الان عضو شوید
spam,اطلاعات کارت بانکی خود را برای دریافت یارانه وارد کنید
spam,هدیه رایگان شما آماده است برای دریافت لینک را باز کنید
spam,عضویت شما در سایت پایان یافت برای تمدید رایگان کلیک کنید
spam,ربح سریع و تضمینی با سرمایه گذاری ارز دیجیتال همین حالا شروع کنید
spam,برنده خودرو صفر شدید برای دریافت کد ملی و شماره کارت ارسال کنید
spam,فرصت استثنایی خرید طلا با ۵۰ درصد تخفیف فقط امروز کلیک کنید
spam,رمز دوم کارت شما منقضی شد برای تغییر فوری وارد لینک شوید
spam,جایزه نقدی شما در انتظار تایید است لینک را لمس کنید
spam,کسب درآمد دلاری از گوشی بدون تجربه ثبت نام رایگان
spam,حساب شما هک شده است فوری رمز خود را در این لینک وارد کنید
spam,شارژ رایگان هدیه شما فقط با یک کلیک دریافت کنید
spam,اعتبار وام شما تایید شد برای دریافت وجه اطلاعات حساب را بفرستید
spam,تخفیف ویژه عید فقط امشب همین حالا خرید کنید برنده شوید
spam,برای جلوگیری از مسدود شدن سیم کارت روی لینک کلیک کنید
ham,سلام فردا ساعت پنج جلسه داریم یادت نره
ham,مامان گفت شام بیا خونه ما
ham,ممنون که کمکم کردی فردا میبینمت
ham,امروز کلاس تعطیل شد استاد بیمار است
ham,کتاب رو فردا میارم دانشگاه
ham,سلام وقت بخیر گزارش را برای شما ایمیل کردم
ham,تولدت مبارک امیدوارم سال خوبی داشته باشی
ham,جلسه به ساعت شش منتقل شد لطفا هماهنگ کن
ham,من رسیدم جلوی در منتظرم
ham,بلیت قطار رو رزرو کردم ساعت هشت صبح حرکت میکنه
ham,نتیجه امتحان اومد نمره ات خوب بود تبریک
ham,امشب فوتبال میبینیم بیا خونه ما
ham,لطفا فایل پروژه را تا پنجشنبه بفرستید
ham,سلام خسته نباشی کار امروز تموم شد
ham,قبض برق این ماه رو پرداخت کردم
ham,فردا صبح با هم صبحانه بخوریم
ham,دکتر وقت من را برای دوشنبه گذاشت
ham,ممنون از دعوتت حتما میام
ham,آقای رضایی تماس گرفتند گفتند بعدا زنگ بزنید
ham,یادت باشه شیر و نون بخری
```

## Acknowledgments

* Inspired by the spam filter and Naive Bayes examples in the Elements of AI / Building AI course
* All example messages were written by me
* Created by aminjmali850000
