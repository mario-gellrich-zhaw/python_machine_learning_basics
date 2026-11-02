"""Load prepared course email features or prepare splits from trusted raw data."""

from email import policy
from email.parser import Parser
import ast
import hashlib
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from enron_features import FEATURE_NAMES


def _decode_message(value):
    if pd.isna(value):
        return '', '', 'missing_message'
    content = str(value)
    status = 'plain_text'
    if content.lstrip().startswith(('[', '"', "'")):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', SyntaxWarning)
                parsed = ast.literal_eval(content)
            if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
                content = '\n'.join(parsed)
                status = 'decoded_list'
            elif isinstance(parsed, str):
                content = parsed
                status = 'decoded_string'
        except (ValueError, SyntaxError, TypeError, RecursionError):
            status = 'literal_parse_fallback'
    has_headers = re.match(r'(?i)^(?:subject|from|to|date|message-id|mime-version|content-type):', content.lstrip())
    if not has_headers:
        return '', content, status
    try:
        message = Parser(policy=policy.default).parsestr(content.lstrip())
        parts = message.walk() if message.is_multipart() else [message]
        bodies = []
        for part in parts:
            if part.get_content_type() == 'text/plain' and part.get_content_disposition() != 'attachment':
                payload = part.get_payload(decode=True)
                bodies.append(payload.decode(part.get_content_charset() or 'utf-8', errors='replace')
                              if isinstance(payload, bytes) else str(part.get_payload()))
        return str(message.get('Subject', '')), '\n'.join(bodies), 'parsed_email'
    except (ValueError, TypeError, LookupError, UnicodeError):
        return '', content, 'email_parse_fallback'


def load_email_splits(source_path, max_messages=30_000, seed=7):
    """Keep reference CSV splits fixed; use the seed only when preparing raw emails."""
    columns = pd.read_csv(source_path, nrows=0).columns
    if set(FEATURE_NAMES).issubset(columns):
        required = {'label', 'split', 'source_row', 'message_key'}
        if not required.issubset(columns):
            raise ValueError('Prepared email data needs label, split, source_row and message_key columns.')
        frame = pd.read_csv(source_path)
        if max_messages is not None and max_messages < len(frame):
            raise ValueError('The reference CSV has fixed splits; use MAX_MESSAGES=30000 or None.')
        if not frame['label'].isin([0, 1]).all() or not frame['split'].isin(['train', 'validation', 'test']).all():
            raise ValueError('Invalid labels or split names in prepared email data.')
        if not frame['message_key'].is_unique or frame['message_key'].isna().any():
            raise ValueError('Prepared email data contains duplicate or missing message keys.')
        if not np.isfinite(frame.loc[:, list(FEATURE_NAMES)].to_numpy(dtype=float)).all():
            raise ValueError('Prepared email features must be finite.')
        splits = tuple(frame.loc[frame['split'].eq(name)].copy() for name in ['train', 'validation', 'test'])
        if any(split.empty for split in splits):
            raise ValueError('Prepared email data must contain all three partitions.')
        return splits
    if not {'message', 'label'}.issubset(columns):
        raise ValueError('Expected prepared course features or raw message and label columns.')
    fixed_text = 'split' in columns
    text_columns = {'message', 'label', 'split', 'source_row', 'message_key'}
    if fixed_text and not text_columns.issubset(columns):
        raise ValueError('Prepared email text needs message, label, split, source_row and message_key columns.')
    raw = pd.read_csv(source_path, usecols=sorted(text_columns) if fixed_text else ['message', 'label'])
    label_values = raw['label'].astype('string').str.strip().str.lower()
    labels = label_values.map({'0': 0, '1': 1, 'ham': 0, 'spam': 1})
    decoded = pd.DataFrame(raw['message'].map(_decode_message).tolist(),
                           columns=['subject', 'body', 'parse_status'], index=raw.index)
    emails = decoded.assign(label=labels, source_path=str(source_path), source_row=raw.index)
    if fixed_text:
        if max_messages is not None and max_messages < len(raw):
            raise ValueError('The email text CSV has fixed splits; do not reduce MAX_MESSAGES below its row count.')
        if labels.isna().any() or not raw['split'].isin(['train', 'validation', 'test']).all():
            raise ValueError('Invalid labels or split names in prepared email text.')
        source_rows = raw['source_row']
        if (not pd.api.types.is_integer_dtype(source_rows)
                or not source_rows.ge(0).all() or not source_rows.is_unique):
            raise ValueError('Prepared email text needs unique, nonnegative integer source rows.')
        content = emails['subject'].fillna('') + '\n' + emails['body'].fillna('')
        # Match the prepared keys without pandas-backend-dependent whitespace rules.
        normalized = content.map(
            lambda text: re.sub(r'[ \t\n\r\f]+', ' ', text).strip().casefold())
        keys = normalized.map(lambda content: hashlib.sha256(content.encode('utf-8')).hexdigest())
        if normalized.eq('').any() or not keys.is_unique or not keys.eq(raw['message_key']).all():
            raise ValueError('Prepared email text contains empty, duplicate or mismatched message content.')
        emails['label'] = labels.astype(int)
        emails['source_row'] = raw['source_row']
        emails['message_key'] = keys
        emails['split'] = raw['split']
        splits = tuple(emails.loc[emails['split'].eq(name)].copy() for name in ['train', 'validation', 'test'])
        if any(split.empty for split in splits):
            raise ValueError('Prepared email text must contain all three partitions.')
        return splits
    emails = emails.loc[emails['label'].notna()].copy()
    emails['label'] = emails['label'].astype(int)
    normalized = (emails['subject'].fillna('') + '\n' + emails['body'].fillna('')).str.replace(r'\s+', ' ', regex=True).str.strip().str.casefold()
    emails['message_key'] = normalized.map(lambda content: hashlib.sha256(content.encode('utf-8')).hexdigest())
    emails = emails.loc[normalized.ne('')].copy()
    label_counts = emails.groupby('message_key')['label'].nunique()
    conflict_keys = label_counts[label_counts > 1].index
    nonconflicting = emails.loc[~emails['message_key'].isin(conflict_keys)]
    unique_emails = nonconflicting.drop_duplicates('message_key').copy()
    if unique_emails['label'].value_counts().min() < 10:
        raise ValueError('Too few examples in one class for stratified splitting.')
    if max_messages is not None and len(unique_emails) > max_messages:
        selected, _ = train_test_split(unique_emails, train_size=max_messages,
                                       stratify=unique_emails['label'], random_state=seed)
    else:
        selected = unique_emails.copy()
    train, heldout = train_test_split(selected, test_size=0.4, stratify=selected['label'], random_state=seed)
    validation, test = train_test_split(heldout, test_size=0.5, stratify=heldout['label'], random_state=seed)
    for left, right in [(train, validation), (train, test), (validation, test)]:
        if not set(left.message_key).isdisjoint(right.message_key):
            raise ValueError('Email partitions overlap.')
    return train, validation, test