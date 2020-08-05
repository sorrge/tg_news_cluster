//    Copyright 2020 sorrge
//
//    This file is part of tg_news_cluster.
//
//    tg_news_cluster is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    tg_news_cluster is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with tg_news_cluster.  If not, see <https://www.gnu.org/licenses/>.
extern crate whatlang;
extern crate libc;

use libc::c_char;
use whatlang::{detect};
use std::ffi::CStr;
use std::ffi::CString;

#[no_mangle]
pub extern fn whatlang_detect(text: *const c_char, confidence: *mut f64) -> *mut c_char {
    unsafe {
        let info = detect(CStr::from_ptr(text).to_str().unwrap());
        if info.is_none() {
            *confidence = 0.0;
            CString::new("None").unwrap().into_raw()
        }
        else {
            *confidence = info.unwrap().confidence();
            CString::new(info.unwrap().lang().code()).unwrap().into_raw()
        }
    }
}


#[no_mangle]
pub extern "C" fn lang_code_free(s: *mut c_char) {
    unsafe {
        if s.is_null() {
            return;
        }

        CString::from_raw(s)
    };
}