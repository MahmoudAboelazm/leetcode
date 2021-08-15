var lengthOfLongestSubstring = function (s) {
  let max = 0,
    tmpLen = 0,
    tmp = "",
    currentI = 0;
  for (let i = currentI; i < s.length; i++) {
    let char = s[i];
    if (tmp.indexOf(char) > -1) {
      if (tmpLen > max) max = tmpLen;
      i = currentI++;
      tmp = s[i];
      tmpLen = 1;
      continue;
    }
    tmp += char;
    tmpLen++;
  }
  return tmpLen < max ? max : tmpLen;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////// Merge K Sorted Lists ////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

const addToResult = (list, val) => {
  while (list) {
    if (list.next === null) {
      list.next = new ListNode(val);
      return;
    }
    list = list.next;
  }
};
var mergeKLists = function (lists) {
  const newList = [];
  for (let i = 0; i < lists.length; i++) {
    let currentNode = lists[i];
    while (currentNode) {
      newList.push(currentNode.val);
      currentNode = currentNode.next;
    }
  }
  newList.sort((a, b) => a - b);
  let result = new ListNode(0);
  newList.map((n) => addToResult(result, n));
  return result.next;
};
//////// Second solution /////////
const addToResult = (list, val) => {
  while (list) {
    if (
      val === list.val ||
      (val > list.val && list.next && val < list.next.val)
    ) {
      list.next = new ListNode(val, list.next);
      return;
    } else if (
      (val < list.val && list.next && val > list.next.val) ||
      val < list.val
    ) {
      list.next = new ListNode(list.val, list.next);
      list.val = val;
      return;
    } else if (list.next === null) {
      list.next = new ListNode(val);
      return;
    }
    list = list.next;
  }
};
var mergeKLists = function (lists) {
  let result;
  for (let i = 0; i < lists.length; i++) {
    let currentNode = lists[i];
    while (currentNode) {
      if (result) {
        addToResult(result, currentNode.val);
      }
      if (!result) {
        result = new ListNode(lists[i].val);
      }
      currentNode = currentNode.next;
    }
  }
  return result ? result : new ListNode(0).next;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////// Find substring /////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

const wordFoundInArr = (word, arr) => {
  let idx = arr.indexOf(word),
    count = 0;
  while (idx > -1) {
    count++;
    idx = arr.indexOf(word, idx + 1);
  }
  return count;
};

const findSubstring = function (s, words) {
  let wLen = words[0].length,
    sLen = s.length,
    wordsLen = words.length,
    currentI = 0,
    map = {},
    result = [];

  if (sLen === wordsLen) {
    let check = wordFoundInArr(s[0], s.split(""));
    if (check === sLen) {
      if (s === words.join("")) {
        return [0];
      } else {
        return [];
      }
    }
  }

  while (currentI < sLen) {
    let nArr = [];

    for (let i = currentI; i < sLen; i = i + wLen) {
      let cWord = "";

      for (let k = i; k < i + wLen; k++) {
        let char = s[k];
        if (char) cWord += char;
      }

      if (!map[cWord]) {
        map[cWord] = wordFoundInArr(cWord, words);
      }
      let wordsWordC = map[cWord];
      if (wordsWordC > 0) {
        let nArrWordC = wordFoundInArr(cWord, nArr);

        if (nArrWordC < wordsWordC) {
          nArr = [...nArr, cWord];
        } else {
          break;
        }
        if (nArr.length === wordsLen) {
          result = [...result, currentI];
          break;
        }
      } else {
        break;
      }
    }
    currentI++;
  }
  return result;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////// Sudoku Solver ///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const isValid = (board, row, col, k) => {
  for (let i = 0; i < 9; i++) {
    const m = 3 * Math.floor(row / 3) + Math.floor(i / 3);
    const n = 3 * Math.floor(col / 3) + (i % 3);
    if (board[row][i] == k || board[i][col] == k || board[m][n] == k) {
      return false;
    }
  }
  return true;
};

const solveSudoku = (board) => {
  for (let row = 0; row < 9; row++) {
    for (let col = 0; col < 9; col++) {
      if (board[row][col] == ".") {
        for (let fill = 1; fill <= 9; fill++) {
          if (isValid(board, row, col, fill)) {
            board[row][col] = `${fill}`;

            if (solveSudoku(board)) return true;

            board[row][col] = ".";
          }
        }

        return false;
      }
    }
  }
  return true;
};
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////// Trapping Rain Water ////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const findTheBiggerNum = (arr, start) => {
  let val = arr[start],
    index = Math.abs(start + 1);
  for (let i = start; i < arr.length; i++) {
    if (arr[i] > val) {
      return i;
    }
    if (i !== start && arr[i] >= arr[index]) index = i;
  }
  return index >= start ? index : null;
};
const check = (idx, height) => {
  let rain = 0;
  idx.forEach((current, i) => {
    let end = idx[i + 1];
    if (end) {
      let smallest;
      if (height[current] > height[end]) {
        smallest = height[end];
      } else {
        smallest = height[current];
      }

      for (let k = current; k < end; k++) {
        if (smallest - height[k] > -1) {
          rain += smallest - height[k];
        }
      }
    }
  });
  return rain;
};

const trap = (height) => {
  let idx = [0];
  let i = 0;

  while (i < height.length) {
    const bigest = findTheBiggerNum(height, i);
    if (bigest) {
      if (idx.indexOf(bigest) < 0) idx.push(bigest);
      i = bigest;
      continue;
    }
    i++;
  }
  return check(idx, height);
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////// Is Match ////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const isMatch = (string, pattern) => {
  // early return when pattern is empty
  if (!pattern) {
    // returns true when string and pattern are empty
    // returns false when string contains chars with empty pattern
    return !string;
  }

  // check if the current char of the string and pattern match when the string has chars
  const hasFirstCharMatch =
    Boolean(string) && (pattern[0] === "." || pattern[0] === string[0]);

  // track when the next character * is next in line in the pattern
  if (pattern[1] === "*") {
    // if next pattern match (after *) is fine with current string, then proceed with it (s, p+2).  That's because the current pattern may be skipped.
    // otherwise check hasFirstCharMatch. That's because if we want to proceed with the current pattern, we must be sure that the current pattern char matches the char
    // If hasFirstCharMatch is true, then do the recursion with next char and current pattern (s+1, p).  That's because current char matches the pattern char.
    return (
      isMatch(string, pattern.slice(2)) ||
      (hasFirstCharMatch && isMatch(string.slice(1), pattern))
    );
  }

  // now we know for sure that we need to do 2 simple actions
  // check the current pattern and string chars
  // if so, then can proceed with next string and pattern chars (s+1, p+1)
  return hasFirstCharMatch ? isMatch(string.slice(1), pattern.slice(1)) : false;
};
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////// Get Permutation //////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const getPermutation = function (n, k) {
  const num = Array.from({ length: n }, (_, i) => i + 1);
  const factorial = [1];
  const result = [];

  for (let i = 1; i <= num.length; i++) {
    factorial.push(factorial[i - 1] * num[i - 1]);
  }

  while (num.length) {
    const idx = Math.ceil(k / factorial[--n]) - 1;

    result.push(num.splice(idx, 1));
    k = k % factorial[n] || factorial[n];
  }

  return result.join("");
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////// Vaild Number //////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const isNumber = function (s) {
  if (s == "Infinity" || s == "-Infinity" || s == "+Infinity") return false;
  return Number(s).toString() !== "NaN";
};

/////////////////// Second Solution ///////////////////////
const isNumber = (s) => !isNaN(s) && s.indexOf("Infinity") === -1;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////// Remove Nth Node From End of List ///////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const getLen = (head) => {
  let len = 0;
  while (head) {
    head = head.next;
    len++;
  }
  return len;
};
const remove = (head, idx, nodeIdx) => {
  let prev = head;
  while (head) {
    if (nodeIdx === idx) {
      prev.next = head.next;
      head.next = prev;
      return;
    }
    nodeIdx++;
    prev = head;
    head = head.next;
  }
};
var removeNthFromEnd = function (head, n) {
  const len = getLen(head);
  let currentIdx = len - (n - 1);
  let nodeIdx = 1;

  if (nodeIdx === currentIdx) return head.next;

  if (len < 2) return head.next;

  remove(head, currentIdx, nodeIdx);

  return head;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////// 4Sum ///////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

var fourSum = function (nums, target) {
  if (nums.length < 4) return [];
  nums.sort((a, b) => a - b);
  let res = [];
  let len = nums.length;

  for (let i = 0; i < len - 3; i++) {
    for (let j = i + 1; j < len - 2; j++) {
      let low = j + 1,
        high = len - 1;

      while (low < high) {
        let sum = nums[i] + nums[j] + nums[high] + nums[low];

        if (sum === target) {
          res.push([nums[i], nums[j], nums[high], nums[low]]);

          while (nums[low] === nums[low + 1]) low++;

          while (nums[high] === nums[high - 1]) high--;

          low++;
          high--;
        } else if (sum > target) {
          high--;
        } else {
          low++;
        }
      }

      while (nums[j] === nums[j + 1]) j++;
    }
    while (nums[i] === nums[i + 1]) i++;
  }

  return res;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////// Letter Combinations //////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

var letterCombinations = function (digits) {
  const numToLetter = {
    2: ["a", "b", "c"],
    3: ["d", "e", "f"],
    4: ["g", "h", "i"],
    5: ["j", "k", "l"],
    6: ["m", "n", "o"],
    7: ["p", "q", "r", "s"],
    8: ["t", "u", "v"],
    9: ["w", "x", "y", "z"],
  };
  let array = [];
  for (let i = digits.length - 1; i >= 0; i--) {
    const num = digits[i];
    if (array.length === 0) {
      array = [...numToLetter[num]];
    } else {
      const newArray = [];
      for (let j of numToLetter[num]) {
        for (let k of array) {
          newArray.push(j + k);
        }
      }
      array = newArray;
    }
  }
  return array;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////// Integer To Roman ///////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const symbolToValue = {
  M: 1000,
  CM: 900,
  D: 500,
  CD: 400,
  C: 100,
  XC: 90,
  L: 50,
  XL: 40,
  X: 10,
  IX: 9,
  V: 5,
  IV: 4,
  I: 1,
};

var intToRoman = function (num) {
  let finalStr = "";
  for (let s in symbolToValue) {
    const v = symbolToValue[s];
    while (num >= v) {
      num -= v;
      finalStr += s;
    }
  }
  return finalStr;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////// Container With Most Water /////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

var maxArea = function (height) {
  let max = 0,
    maxIdx = height.length - 1,
    minIdx = 0;

  while (minIdx < maxIdx) {
    // Find how many steps between the current index and the maxOne
    let steps = maxIdx - minIdx;

    // Get two numbers one from the beginning of the array and the other one from the end of the array
    let numOne = height[minIdx],
      numTwo = height[maxIdx];

    let smallest = Math.min(numOne, numTwo);

    max = Math.max(max, smallest * steps);

    // If the end number is bigger than the number from the beginning then we will go forward else we will go backward
    numTwo > numOne ? minIdx++ : maxIdx--;
  }

  return max;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////// Swap Nodes in Pairs //////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

const makeSwap = (head) => {
  let swap = true;
  while (head) {
    if (swap && head.next) {
      let tmp = head.val;
      head.val = head.next.val;
      head.next.val = tmp;
      swap = false;
      head = head.next;
      continue;
    }
    head = head.next;
    swap = true;
  }
  return;
};
var swapPairs = function (head) {
  makeSwap(head);
  return head;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////// Search in Rotated Sorted Array /////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

var search = (nums, target) => nums.indexOf(target);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////// Find First and Last Position of Element in Sorted Array ///////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

var searchRange = function (nums, target) {
  let currentIdx = nums.indexOf(target);

  if (currentIdx === -1) return [-1, -1];

  let res = [currentIdx],
    last = currentIdx;

  while (currentIdx > -1) {
    currentIdx = nums.indexOf(target, currentIdx + 1);
    if (currentIdx > -1) last = currentIdx;
  }

  res = [...res, last];
  return res;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////// Rotate Image ///////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

var rotate = function (matrix) {
  let rowLen = matrix.length - 1;

  matrix.reverse();

  for (let row = 0; row <= rowLen; row++) {
    for (let col = 0; col < row; col++) {
      let tmp = matrix[row][col];
      matrix[row][col] = matrix[col][row];
      matrix[col][row] = tmp;
    }
  }

  return matrix;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////// Largest Rectangle Area ////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

var largestRectangleArea = function (heights) {
  // add 0 to the end of heights
  heights.push(0);
  // create stack starting with value -1
  const stack = [-1];
  // create max variable
  let max = 0;
  // loop through bars in histogram
  for (let i = 0; i < heights.length; i++) {
    // while current bar is less than height of index at top of stack
    while (heights[i] < heights[stack[stack.length - 1]]) {
      // h is the height @ index at top of stack
      let h = heights[stack.pop()];
      // w is i minus index @ top of stack - 1
      let w = i - stack[stack.length - 1] - 1;
      // max is the max area calculated thus far
      max = Math.max(max, h * w);
    }
    // push i to the stack
    stack.push(i);
  }

  return max;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////// Reverse Nodes in k-Group ///////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

const makeReverse = (list) => {
  let prev = null,
    next;
  while (list) {
    next = list.next;
    list.next = prev;
    prev = list;
    list = next;
  }
  list = prev;
  return list;
};

const add = (list, val) => {
  while (list) {
    if (list.next === null) {
      list.next = new ListNode(val);
      return;
    }
    list = list.next;
  }
};
const merge = (list, list2) => {
  while (list) {
    if (list.next === null) {
      list.next = list2;
      return;
    }
    list = list.next;
  }
};

const getLen = (list) => {
  let len = 0;
  while (list) {
    len++;
    list = list.next;
  }
  return len;
};
var reverseKGroup = function (head, k) {
  const l = getLen(head);
  if (k === l) return makeReverse(head);

  let idx = 0,
    reversedList = new ListNode(0),
    result;
  while (head) {
    add(reversedList, head.val);
    idx++;
    if (idx === k) {
      const list = makeReverse(reversedList.next);
      result ? merge(result, list) : (result = list);
      idx = 0;
      reversedList = new ListNode(0);
    }
    head = head.next;
  }
  merge(result, reversedList.next);
  return result;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////// Maximal Rectangle ///////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

const maximalRectangle = function (matrix) {
  if (!matrix.length) return 0;
  const ROWS = matrix.length;
  const COLS = matrix[0].length;
  const dp = Array.from({ length: ROWS }, () => Array(COLS).fill(0));

  let maxArea = 0;

  for (let row = 0; row < ROWS; row++) {
    for (let col = 0; col < COLS; col++) {
      //update height
      if (row === 0) dp[row][col] = matrix[row][col] == "1" ? 1 : 0;
      else dp[row][col] = matrix[row][col] == "1" ? dp[row - 1][col] + 1 : 0;

      //update area
      let minHeight = dp[row][col];

      for (let pointer = col; pointer >= 0; pointer--) {
        if (minHeight === 0) break;
        minHeight = Math.min(minHeight, dp[row][pointer]);
        maxArea = Math.max(maxArea, minHeight * (col - pointer + 1));
      }
    }
  }
  return maxArea;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////// Distinct Subsequences /////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

var numDistinct = function (s, t) {
  //Create DP array containig possible results for each char in t
  let dp = Array(t.length + 1).fill(0);
  //Base case - empty string 1 result
  dp[0] = 1;

  //Iterate s string
  for (let i = 0; i < s.length; i++) {
    //Iterate t string (from end to start so we don't process data that we've just entered)
    for (let j = t.length - 1; j >= 0; j--) {
      //Char match
      if (s[i] === t[j]) {
        //Add this char count to next position
        dp[j + 1] += dp[j];
      }
    }
  }
  return dp[t.length];
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////// Binary Tree Maximum Path Sum //////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

var maxPathSum = function (root) {
  let maxSum = -Infinity;

  function getMaxPathSum(root) {
    if (!root) {
      return 0;
    }
    const leftPathSum = Math.max(getMaxPathSum(root.left), 0),
      rightPathSum = Math.max(getMaxPathSum(root.right), 0);

    maxSum = Math.max(maxSum, root.val + leftPathSum + rightPathSum);
    return Math.max(root.val + leftPathSum, root.val + rightPathSum);
  }

  getMaxPathSum(root);
  return maxSum;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////// Invert Binary Tree ///////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
var invertTree = function (root) {
  if (!root || (!root.left && !root.right)) return root;

  let tmp = root.left;
  root.left = root.right;
  root.right = tmp;

  invertTree(root.left);
  invertTree(root.right);

  return root;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////// Longest Palindromic Substring //////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

const checkCharCount = (s, char) => {
  let idx = s.indexOf(char),
    count = 0;
  while (idx > -1) {
    idx = s.indexOf(char, idx + 1);
    count++;
  }
  return count;
};

const checkTmp = (word) => {
  let len = word.length - 1;
  let l = len / 2;
  for (let i = 0; i <= l; i++) {
    if (word[i] !== word[len - i]) return false;
  }
  return true;
};

var longestPalindrome = function (s) {
  let word = "",
    sLen = s.length,
    charCount = checkCharCount(s, s[0]);
  if (charCount === sLen) return s;
  for (let i = 0; i < sLen; i++) {
    let tmp = "";
    for (let k = i; k < sLen; k++) {
      tmp += s[k];
      if (checkTmp(tmp) && tmp.length > word.length) word = tmp;
    }
  }
  return word;
};

///////////// Second Solution ///////////////

var longestPalindromee = function (s) {
  let subStr = "";
  // inside the for loop we will check if it satisfy 2 condition (odd or even)
  for (let i = 0; i < s.length; i++) {
    let left = i;
    let right = i;

    // odd length
    while (left >= 0 && right < s.length && s[left] === s[right]) {
      if (right - left + 1 > subStr.length) {
        subStr = s.substring(left, right + 1);
      }
      left--;
      right++;
    }
    // even length
    left = i;
    right = i + 1;
    while (left >= 0 && right < s.length && s[left] === s[right]) {
      if (right - left + 1 > subStr.length) {
        subStr = s.substring(left, right + 1);
      }
      left--;
      right++;
    }
  }
  return subStr;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////// Longest Common Prefix ///////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

var longestCommonPrefix = function (strs) {
  strs.sort();
  for (let i = 0; i < strs[0].length; i++) {
    if (strs[0][i] !== strs[strs.length - 1][i]) return strs[0].substr(0, i);
  }
  return strs[0];
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////// Valid Sudoku /////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////

var isValidSudoku = function (board) {
  let map = new Map();
  for (let row = 0; row < 9; row++) {
    for (let col = 0; col < 9; col++) {
      let char = board[row][col];
      if (char !== ".") {
        let box = Math.floor(row / 3) + "-" + Math.floor(col / 3) + "-" + char;
        let charRow = char + "row" + row,
          charCol = char + "col" + col;
        if (map.has(box) || map.has(charRow) || map.has(charCol)) return false;
        map.set(box, true);
        map.set(charRow, true);
        map.set(charCol, true);
      }
    }
  }
  return true;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////// Multiply Strings ///////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////

const multiply = (num1, num2) => (BigInt(num1) * BigInt(num2)).toString();

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////// Jump Game II /////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////

var jump = function (nums) {
  let len = nums.length - 1,
    curr = -1,
    next = 0,
    ans = 0;
  for (let i = 0; next < len; i++) {
    if (i > curr) ans++, (curr = next);
    next = Math.max(next, nums[i] + i);
  }
  return ans;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////// Group Anagrams ///////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////

var groupAnagrams = function (strs) {
  let map = new Map();
  for (let anagram of strs) {
    const sorted_anagram = anagram.split("").sort().join("");
    if (!map.has(sorted_anagram)) map.set(sorted_anagram, []);
    map.get(sorted_anagram).push(anagram);
  }
  return [...map.values()];
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////// Spiral Order ////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////

var spiralOrder = function (matrix) {
  const result = [];
  let row_start = 0,
    row_end = matrix.length - 1,
    col_start = 0,
    col_end = matrix[0].length - 1,
    i;

  while (row_start <= row_end && col_start <= col_end) {
    for (i = col_start; i <= col_end; i++) {
      result.push(matrix[row_start][i]);
    }
    row_start++;

    for (i = row_start; i <= row_end; i++) {
      result.push(matrix[i][col_end]);
    }
    col_end--;

    if (row_start <= row_end) {
      for (i = col_end; i >= col_start; --i) {
        result.push(matrix[row_end][i]);
      }
      --row_end;
    }

    if (col_start <= col_end) {
      for (i = row_end; i >= row_start; --i) {
        result.push(matrix[i][col_start]);
      }
      ++col_start;
    }
  }

  return result;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////// Jump Game /////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////

var canJump = function (nums) {
  let len = nums.length;
  if (len === 1) return true;

  let max = 0;

  for (let i = 0; i < len; i++) {
    max = Math.max(max, i + nums[i]);
    if (max === len - 1) return true;
    if (i >= max) return false;
  }

  return true;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////// Find Minimum in Rotated Sorted Array II /////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////

var findMin = function (nums) {
  let start = 0,
    end = nums.length - 1;

  while (nums[start] == nums[end] && start < end) {
    start++;
  }

  while (start < end) {
    let mid = Math.floor((end + start) / 2);

    if (nums[start] < nums[end]) return nums[start];

    nums[mid] >= nums[start] ? (start = mid + 1) : (end = mid);
  }

  return nums[start];
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////// Reverse Linked List II ///////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

var reverseBetween = function (head, left, right) {
  const merge = (list, list2) => {
    while (list) {
      if (list.next === null) {
        list.next = list2;
        return;
      }
      list = list.next;
    }
  };
  const addToList = (list, val) => {
    while (list) {
      if (list.next === null) {
        list.next = new ListNode(val);

        return;
      }
      list = list.next;
    }
  };
  const addTo = (start, end, val, left, nodeIdx) =>
    nodeIdx < left ? addToList(start, val) : addToList(end, val);

  const makeReverse = (list, left, right) => {
    let prev = null,
      nodeIdx = 1,
      next;
    let start = new ListNode(0),
      end = new ListNode(0);
    while (list) {
      next = list.next;

      if (nodeIdx >= left && nodeIdx <= right) {
        list.next = prev;
        prev = list;
      } else {
        addTo(start, end, list.val, left, nodeIdx);
      }

      list = next;
      nodeIdx++;
    }

    if (start.next !== null) {
      merge(start.next, prev);
      merge(start, end.next);
      return start.next;
    }

    if (end.next !== null) merge(prev, end.next);

    return prev;
  };
  return makeReverse(head, left, right);
};
